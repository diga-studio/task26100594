"""PyForge 批量发布链路 —— 依赖图组件。

节点是 block，边 u -> v 表示 u 依赖 v。本组件只维护当前依赖关系；
版本、权限及其他发布字段由调用方验证。

内部不变式：维护一个拓扑序 rank，使每条已提交边 u -> v 都满足
rank[u] > rank[v]（被依赖方排在更前）。该序只用于加速成环检查，对外不可见；
图的任何拓扑序都同样有效，因此拒绝批次后无需还原 rank。

增量成环检查（Pearce-Kelly 风格）：
- 新边 u -> v 若满足 rank[u] > rank[v]，与现有拓扑序一致，不可能成环，O(1) 接受；
- 否则只需在 rank 区间 [rank[u], rank[v]] 内从 v 前向搜索 u：
  任何 v ⇝ u 的路径 rank 严格递减，必然落在该区间内；
  找不到 u 时，把区间内"可达 v 的"与"被 v 可达的"节点整体重排即可恢复不变式。
搜索与重排只触及受影响区间，不扫描全图、不复制全部依赖、不保存全图副本。
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Set, Tuple, Union

NodeId = Hashable
Cycle = List[NodeId]

MAX_BATCH_NODES = 100
MAX_DIRECT_DEPS = 15


class DependencyGraph:
    """固定节点集合上的依赖图（始终保持无环）。

    replace_batch 把批次中所有替换同时作用于旧图得到候选图，
    当且仅当候选图无环时整批生效；否则返回候选图中的一个真实环，
    已提交依赖与后续判断与本批从未发生时一致。
    """

    __slots__ = ("_deps", "_users", "_order", "_rank", "edge_checks")

    def __init__(self, graph: Mapping[NodeId, Iterable[NodeId]]) -> None:
        # _deps[u]: u 直接依赖的节点集合（出边）；_users[v]: 直接依赖 v 的节点集合（入边）
        self._deps: Dict[NodeId, Set[NodeId]] = {}
        self._users: Dict[NodeId, Set[NodeId]] = {}
        self._order: List[NodeId] = []          # 拓扑序：下标即 rank
        self._rank: Dict[NodeId, int] = {}
        # 公开计数器：组件自上次清零以来检查过的依赖边条数（含重复检查、
        # 含增删边与前向/反向搜索中对辅助入边结构的检查）。
        self.edge_checks: int = 0

        for node in graph:
            if node in self._deps:
                raise ValueError(f"重复的节点: {node!r}")
            self._deps[node] = set()
            self._users[node] = set()
        for node, targets in graph.items():
            for t in targets:
                if t not in self._deps:
                    raise KeyError(f"依赖目标不存在: {t!r}")
                if t == node:
                    raise ValueError(f"初始图存在自环: {node!r}")
                if t not in self._deps[node]:
                    self._deps[node].add(t)
                    self._users[t].add(node)

        # Kahn 拓扑排序（迭代实现，长链不会溢出调用栈）：
        # 每次取出一个"没有未处理依赖"的节点，被依赖方排在更前。
        remaining = {n: len(d) for n, d in self._deps.items()}
        queue = deque(n for n, c in remaining.items() if c == 0)
        order = self._order
        while queue:
            x = queue.popleft()
            self._rank[x] = len(order)
            order.append(x)
            for u in self._users[x]:
                remaining[u] -= 1
                if remaining[u] == 0:
                    queue.append(u)
        if len(order) != len(self._deps):
            leftover = [n for n, c in remaining.items() if c > 0]
            raise ValueError(f"初始图存在环，涉及节点（示例）: {leftover[:5]!r}")

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def dependencies(self, node: NodeId) -> List[NodeId]:
        """返回该节点当前已提交的直接依赖（每次返回新列表，调用方修改不影响组件）。"""
        if node not in self._deps:
            raise KeyError(f"节点不存在: {node!r}")
        return list(self._deps[node])

    # ------------------------------------------------------------------
    # 内部：单条边的增删与增量成环检查
    # ------------------------------------------------------------------

    def _attach(self, u: NodeId, v: NodeId) -> None:
        self.edge_checks += 1
        self._deps[u].add(v)
        self._users[v].add(u)

    def _detach(self, u: NodeId, v: NodeId) -> None:
        self.edge_checks += 1
        self._deps[u].discard(v)
        self._users[v].discard(u)

    def _add_edge_checked(self, u: NodeId, v: NodeId) -> Optional[Cycle]:
        """向当前图加边 u -> v。成环则不加边并返回环；否则加边并维护拓扑序。"""
        if u == v:
            return [u, u]
        rank = self._rank
        lo, hi = rank[u], rank[v]
        if lo > hi:
            # 与拓扑序一致：任何 v ⇝ u 的路径都会使 rank 严格递减，
            # 即要求 rank[v] > rank[u]，矛盾，故不可能成环。
            self._attach(u, v)
            return None

        # 前向搜索：从 v 沿出边在 rank 区间 [lo, hi] 内找 u。
        # 路径上 rank 严格递减，故区间外的节点不可能在 v ⇝ u 的路径上。
        parent: Dict[NodeId, Optional[NodeId]] = {v: None}
        stack = [v]
        found = False
        while stack and not found:
            x = stack.pop()
            for y in self._deps[x]:
                self.edge_checks += 1
                ry = rank[y]
                if ry < lo or ry > hi or y in parent:
                    continue
                parent[y] = x
                if y == u:
                    found = True
                    break
                stack.append(y)

        if found:
            # 还原路径 v ⇝ u，拼成环 u -> v -> ... -> u。
            path = [u]
            while path[-1] != v:
                path.append(parent[path[-1]])
            return [u] + path[::-1]

        # 无环：重排区间恢复不变式。
        # delta_f：区间内 v 可达的节点（含 v），必须排到 u 之前；
        # delta_b：区间内可达 u 的节点（含 u），必须排到 v 之后。
        # 二者不相交，否则 v ⇝ w ⇝ u 与"u 不可达"矛盾。
        delta_b: Set[NodeId] = {u}
        stack = [u]
        while stack:
            x = stack.pop()
            for w in self._users[x]:
                self.edge_checks += 1
                rw = rank[w]
                if rw < lo or rw > hi or w in delta_b:
                    continue
                delta_b.add(w)
                stack.append(w)

        # 按旧 rank 排序后，delta_f 整体占低槽位、delta_b 整体占高槽位。
        positions = sorted(rank[w] for w in list(parent) + list(delta_b))
        front = sorted(parent, key=rank.__getitem__)
        back = sorted(delta_b, key=rank.__getitem__)
        for w, p in zip(front + back, positions):
            self._order[p] = w
            rank[w] = p

        self._attach(u, v)
        return None

    # ------------------------------------------------------------------
    # 批量替换
    # ------------------------------------------------------------------

    def replace_batch(self, changes) -> Optional[Cycle]:
        """原子地替换一批节点的完整直接依赖列表；空列表表示清除依赖。

        changes 可以是 {节点: 依赖列表} 或 [(节点, 依赖列表), ...]。
        成功返回 None；失败返回候选图中的一个环（首尾相同、中间节点不重复、
        每条边都真实存在于候选图），此时已提交依赖保持不变。
        """
        items = list(changes.items()) if hasattr(changes, "items") else list(changes)
        if len(items) > MAX_BATCH_NODES:
            raise ValueError(f"批次节点数超过 {MAX_BATCH_NODES}")

        # 归一化：去重、校验。结果只取决于"每个节点的最终依赖列表"，
        # 与批次条目的排列无关。
        plan: Dict[NodeId, List[NodeId]] = {}
        for node, targets in items:
            if node not in self._deps:
                raise KeyError(f"节点不存在: {node!r}")
            new_deps = list(dict.fromkeys(targets))
            if len(new_deps) > MAX_DIRECT_DEPS:
                raise ValueError(f"节点 {node!r} 的直接依赖超过 {MAX_DIRECT_DEPS} 条")
            for t in new_deps:
                if t not in self._deps:
                    raise KeyError(f"依赖目标不存在: {t!r}")
            plan[node] = new_deps

        # 计算每个节点的边差分：只动真正变化的边。
        to_delete: List[Tuple[NodeId, NodeId]] = []
        to_add: List[Tuple[NodeId, NodeId]] = []
        for node, new_deps in plan.items():
            old = self._deps[node]
            new_set = set(new_deps)
            for v in old - new_set:
                to_delete.append((node, v))
            for v in new_deps:
                if v not in old:
                    to_add.append((node, v))

        # 第一阶段：先删。删边不会成环；先删可避免"边稍后会被删掉"
        # 的旧边参与成环判断，保证只由候选图决定结果。
        for u, v in to_delete:
            self._detach(u, v)

        # 第二阶段：逐条加边并做增量成环检查。
        # 中间图始终是候选图的子图：若候选图无环，任何中间图都无环，
        # 检查不会误报；若候选图有环，环上最后一条新边加入时，
        # 环的其余边已全部就位，必然被检出。因此接受与否只取决于候选图。
        added: List[Tuple[NodeId, NodeId]] = []
        for u, v in to_add:
            cycle = self._add_edge_checked(u, v)
            if cycle is not None:
                self._rollback(to_delete, added)
                return cycle
            added.append((u, v))
        return None

    def _rollback(
        self,
        to_delete: List[Tuple[NodeId, NodeId]],
        added: List[Tuple[NodeId, NodeId]],
    ) -> None:
        """撤销本批已做的全部边修改，恢复到批次开始前的已提交图。

        只按操作日志反向撤销（最多批次规模的边），不保存整图副本。
        拓扑序 rank 在失败批次中可能已被重排，但它仍是旧图的有效拓扑序
        （边更少的图继承原序的正确性），无需还原。
        """
        for u, v in added:
            self._detach(u, v)
        for u, v in to_delete:
            cycle = self._add_edge_checked(u, v)
            # 恢复旧边时中间图始终是旧图的子图，旧图无环，故不可能成环。
            assert cycle is None, "回滚恢复旧边时不应成环"
