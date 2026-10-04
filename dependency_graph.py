"""PyForge 批量发布链路 —— 依赖图组件。

节点是 block，边 A -> B 表示 A 依赖 B。组件只维护当前依赖关系；
版本、权限等发布字段由调用方负责验证。

核心设计：
- 每个节点维护一个拓扑级别 rank，不变式：对每条边 a -> b 都有 rank[a] < rank[b]。
- replace_batch 先把本批所有旧边删除（删除不会成环），再逐条加入新边；
  加边时用 Pearce & Kelly 增量拓扑算法，只在受影响的 rank 区间内做搜索，
  发现环则按撤销日志回滚本批全部改动（边与 rank），并重构出真实的环路径。
- 所有搜索均为迭代式，深度接近 10 万的链不会撑爆调用栈。
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Set, Tuple

Node = Hashable

MAX_BATCH = 100
MAX_OUT_DEGREE = 15


class DependencyGraph:
    """固定节点集合上的有向依赖图，支持原子批量替换与增量环检测。"""

    def __init__(
        self,
        nodes: Iterable[Node],
        dependencies: Mapping[Node, Iterable[Node]],
    ) -> None:
        """从一个无环图创建组件，做一次全图预处理（拓扑分级）。

        :param nodes: 全部节点（集合固定，之后不增删）。
        :param dependencies: 节点 -> 直接依赖目标，缺省视为空列表。
        :raises ValueError: 目标不存在、出度超限或图中有环。
        """
        self._deps: Dict[Node, Set[Node]] = {n: set() for n in nodes}
        self._users: Dict[Node, Set[Node]] = {n: set() for n in nodes}
        # edge_checks 统计验证与索引维护中检查的依赖边次数（含重复检查），
        # 供性能测试断言使用。
        self.edge_checks: int = 0

        for src, targets in dependencies.items():
            if src not in self._deps:
                raise ValueError(f"未知节点: {src!r}")
            seen: Set[Node] = set()
            for dst in targets:
                if dst not in self._deps:
                    raise ValueError(f"未知依赖目标: {dst!r}")
                if dst in seen:
                    continue
                seen.add(dst)
                self._deps[src].add(dst)
                self._users[dst].add(src)
            if len(seen) > MAX_OUT_DEGREE:
                raise ValueError(f"节点 {src!r} 的直接依赖超过 {MAX_OUT_DEGREE} 个")

        self._rank: Dict[Node, int] = self._topological_ranks()

    def _topological_ranks(self) -> Dict[Node, int]:
        """迭代式 Kahn 算法，给出满足 rank[a] < rank[b]（对每条边 a->b）的分级。"""
        indegree: Dict[Node, int] = {n: 0 for n in self._deps}
        for src, targets in self._deps.items():
            for dst in targets:
                indegree[dst] += 1
        queue: deque[Node] = deque(n for n in self._deps if indegree[n] == 0)
        rank: Dict[Node, int] = {}
        order = 0
        while queue:
            node = queue.popleft()
            rank[node] = order
            order += 1
            for nxt in self._deps[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if len(rank) != len(self._deps):
            raise ValueError("初始图存在环，无法创建组件")
        return rank

    def dependencies(self, node: Node) -> List[Node]:
        """返回节点当前已提交的直接依赖（副本，修改不影响组件）。"""
        if node not in self._deps:
            raise ValueError(f"未知节点: {node!r}")
        return list(self._deps[node])

    def replace_batch(
        self, changes: Mapping[Node, Iterable[Node]]
    ) -> Optional[List[Node]]:
        """原子地替换一批节点的完整直接依赖列表。

        语义：把批次中所有替换同时作用于旧图得到候选图，
        当且仅当候选图无环时接受（返回 None）；否则本批不产生任何
        影响，返回候选图中一个真实存在的环（首尾相同、中间不重复）。

        实现分两阶段：先删除本批涉及节点的全部旧边（删边不会成环），
        再逐条加入新边并做增量环检测。任何时刻发现的环都只由
        “已删除完的旧图 + 部分新边”构成，因而必然存在于最终候选图中。
        """
        items = self._normalize(changes)

        # 撤销日志：只需记录本批触到的节点（<=100 个）与被调过级的节点，
        # 不需要整图副本。
        old_deps: Dict[Node, Set[Node]] = {}
        old_ranks: Dict[Node, int] = {}

        # 阶段一：删除所有被替换节点的旧边。
        for node, _ in items:
            old_deps[node] = set(self._deps[node])
            for dst in old_deps[node]:
                self._remove_edge(node, dst)

        # 阶段二：逐条加入新边，增量检测。
        conflict: Optional[List[Node]] = None
        for node, targets in items:
            for dst in targets:
                cycle = self._add_edge_checked(node, dst, old_ranks)
                if cycle is not None:
                    conflict = cycle
                    break
            if conflict is not None:
                break

        if conflict is None:
            return None

        # 回滚：恢复本批涉及节点的旧依赖与所有被改动的 rank，
        # 使已提交状态与“本批从未发生”完全一致。
        for node, old in old_deps.items():
            for dst in set(self._deps[node]):
                self._remove_edge(node, dst)
            for dst in old:
                self._restore_edge(node, dst)
        for node, rank in old_ranks.items():
            self._rank[node] = rank
        return conflict

    # ---- 内部：批次规范化与校验 ----

    def _normalize(
        self, changes: Mapping[Node, Iterable[Node]]
    ) -> List[Tuple[Node, List[Node]]]:
        if len(changes) > MAX_BATCH:
            raise ValueError(f"单批最多替换 {MAX_BATCH} 个节点")
        items: List[Tuple[Node, List[Node]]] = []
        for node, targets in changes.items():
            if node not in self._deps:
                raise ValueError(f"未知节点: {node!r}")
            dedup: List[Node] = []
            seen: Set[Node] = set()
            for dst in targets:
                if dst not in self._deps:
                    raise ValueError(f"未知依赖目标: {dst!r}")
                if dst not in seen:
                    seen.add(dst)
                    dedup.append(dst)
            if len(dedup) > MAX_OUT_DEGREE:
                raise ValueError(f"节点 {node!r} 的直接依赖超过 {MAX_OUT_DEGREE} 个")
            items.append((node, dedup))
        return items

    # ---- 内部：边的增删（同时维护正向/反向邻接与检查计数） ----

    def _remove_edge(self, src: Node, dst: Node) -> None:
        self.edge_checks += 1
        self._deps[src].discard(dst)
        self._users[dst].discard(src)

    def _restore_edge(self, src: Node, dst: Node) -> None:
        self.edge_checks += 1
        self._deps[src].add(dst)
        self._users[dst].add(src)

    # ---- 内部：Pearce & Kelly 增量拓扑加边 ----

    def _add_edge_checked(
        self, src: Node, dst: Node, old_ranks: Dict[Node, int]
    ) -> Optional[List[Node]]:
        """在（已删完旧边的）当前图中加入 src->dst。

        返回 None 表示安全；否则返回一个真实存在的环。
        安全时按需重排受影响区间的 rank，并把旧 rank 记入 old_ranks。
        """
        if src == dst:
            return [src, src]
        if self._rank[src] < self._rank[dst]:
            # 新边与拓扑序同向，不可能成环，O(1) 通过。
            self._restore_edge(src, dst)
            return None

        upper = self._rank[src]
        lower = self._rank[dst]

        # 前向搜索：从 dst 沿依赖边出发，只访问 rank <= rank[src] 的节点
        # （任何 dst => src 的路径必然落在这个区间内）。
        # 若到达 src，则新边会闭合出一个环，沿途记录父指针以重构路径。
        parent: Dict[Node, Node] = {}
        forward: Set[Node] = {dst}
        stack: List[Node] = [dst]
        hit = False
        while stack and not hit:
            cur = stack.pop()
            for nxt in self._deps[cur]:
                self.edge_checks += 1
                if nxt == src:
                    parent[nxt] = cur
                    hit = True
                    break
                if nxt in forward or self._rank[nxt] > upper:
                    continue
                forward.add(nxt)
                parent[nxt] = cur
                stack.append(nxt)

        if hit:
            # 重构路径 dst => ... => src，环 = src -> dst -> ... -> src。
            path = [src]
            while path[-1] != dst:
                path.append(parent[path[-1]])
            path.reverse()  # dst ... src
            return [src] + path  # src, dst, ..., src

        # 无环：后向搜索 src 的祖先（rank >= rank[dst] 的区间）。
        backward: Set[Node] = {src}
        stack = [src]
        while stack:
            cur = stack.pop()
            for prev in self._users[cur]:
                self.edge_checks += 1
                if prev in backward or self._rank[prev] < lower:
                    continue
                backward.add(prev)
                stack.append(prev)

        # 重排：backward（src 的祖先）整体排在 forward（dst 的后代）之前，
        # 两组各自保持原有相对顺序，复用两组原有的 rank 值。
        back_sorted = sorted(backward, key=lambda n: self._rank[n])
        fwd_sorted = sorted(forward, key=lambda n: self._rank[n])
        pool = sorted(self._rank[n] for n in back_sorted + fwd_sorted)
        for node, new_rank in zip(back_sorted + fwd_sorted, pool):
            if node not in old_ranks:
                old_ranks[node] = self._rank[node]
            self._rank[node] = new_rank

        self._restore_edge(src, dst)
        return None
