"""DependencyGraph 正确性与性能测试。"""
import random
import unittest

from depgraph import DependencyGraph


def candidate_edges(old_graph, changes):
    """模拟"批次同时作用于旧图"，返回候选图的边集合 {(u, v)}。"""
    edges = {(u, v) for u, deps in old_graph.items() for v in deps}
    for u, new_deps in changes.items():
        edges = {e for e in edges if e[0] != u}
        edges.update((u, v) for v in new_deps)
    return edges


def assert_valid_cycle(testcase, cycle, edges):
    """环必须首尾相同、中间节点不重复，且每条边真实存在于候选图。"""
    testcase.assertGreaterEqual(len(cycle), 2)
    testcase.assertEqual(cycle[0], cycle[-1])
    middle = cycle[:-1]
    testcase.assertEqual(len(middle), len(set(middle)))
    for a, b in zip(cycle, cycle[1:]):
        testcase.assertIn((a, b), edges)


class ReplaceVsReverseTests(unittest.TestCase):
    """第一组：同批撤旧边 + 加反向边；菱形共享依赖不误判。"""

    def test_remove_and_reverse_in_same_batch(self):
        # 旧图只有 A -> B；同批清除 A 的依赖并令 B -> A。
        # 逐项作用会暂时成环，但最终图无环，必须成功。
        old = {"A": ["B"], "B": []}
        g = DependencyGraph(old)
        self.assertIsNone(g.replace_batch({"A": [], "B": ["A"]}))
        self.assertEqual(g.dependencies("A"), [])
        self.assertEqual(g.dependencies("B"), ["A"])

    def test_input_order_does_not_matter(self):
        # 交换批次条目顺序，结果一致。
        old = {"A": ["B"], "B": []}
        g1 = DependencyGraph(old)
        g2 = DependencyGraph(old)
        self.assertIsNone(g1.replace_batch([("A", []), ("B", ["A"])]))
        self.assertIsNone(g2.replace_batch([("B", ["A"]), ("A", [])]))
        self.assertEqual(g1.dependencies("B"), g2.dependencies("B"))

    def test_diamond_shared_dependency_not_false_cycle(self):
        # 菱形 A->B,A->C,B->D,C->D，X->A；Y 原依赖 D。
        # 把 Y 改为依赖 X：检查时会从 X 出发两次到达 D（经 B、经 C），
        # 多次到达同一节点不是环，必须成功。
        old = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": [],
               "X": ["A"], "Y": ["D"]}
        g = DependencyGraph(old)
        self.assertIsNone(g.replace_batch({"Y": ["X"]}))
        self.assertEqual(g.dependencies("Y"), ["X"])
        # 菱形结构保持完好
        self.assertEqual(set(g.dependencies("A")), {"B", "C"})
        self.assertEqual(g.dependencies("D"), [])


class LongChainCycleTests(unittest.TestCase):
    """第二组：环必须同时使用两条新边和一段未修改长链。"""

    K = 2000

    def setUp(self):
        # 链 c0 -> c1 -> ... -> c(K-1)，外加孤立节点 X。
        self.old = {f"c{i}": [f"c{i+1}"] for i in range(self.K - 1)}
        self.old[f"c{self.K-1}"] = []
        self.old["X"] = []
        self.last = f"c{self.K-1}"

    def test_two_new_edges_plus_long_chain_cycle_rejected(self):
        g = DependencyGraph(self.old)
        # 新边 X->c0 与 c(K-1)->X，与长链共同构成环；两条新边缺一不可。
        changes = {"X": ["c0"], self.last: ["X"]}
        cycle = g.replace_batch(changes)
        self.assertIsNotNone(cycle)
        edges = candidate_edges(self.old, changes)
        assert_valid_cycle(self, cycle, edges)
        # 环确实穿过未修改的长链（长度远大于 2）
        self.assertGreater(len(cycle), 100)
        # 两条新边都在环上
        pairs = set(zip(cycle, cycle[1:]))
        self.assertIn(("X", "c0"), pairs)
        self.assertIn((self.last, "X"), pairs)
        # 拒绝后已提交依赖与本批从未发生时一致
        self.assertEqual(g.dependencies("X"), [])
        self.assertEqual(g.dependencies(self.last), [])
        self.assertEqual(g.dependencies("c0"), ["c1"])

    def test_no_residual_pollution_after_rejection(self):
        g = DependencyGraph(self.old)
        bad = {"X": ["c0"], self.last: ["X"]}
        self.assertIsNotNone(g.replace_batch(bad))

        # 失败后立即提交合法批次：X -> c500 是前向边，必须成功
        self.assertIsNone(g.replace_batch({"X": ["c500"]}))
        self.assertEqual(g.dependencies("X"), ["c500"])

        # 再提交真正成环的批次：c500 -> X 与 X -> c500 构成环
        cycle = g.replace_batch({"c500": ["X"]})
        self.assertIsNotNone(cycle)
        committed = {n: [d for d in g.dependencies(n)] for n in
                     [f"c{i}" for i in range(self.K)] + ["X"]}
        edges = {(u, v) for u, deps in committed.items() for v in deps}
        edges = {e for e in edges if e[0] != "c500"} | {("c500", "X")}
        assert_valid_cycle(self, cycle, edges)
        # 再次被拒后，c500 仍保持原链边，X 仍是合法批次的结果
        self.assertEqual(g.dependencies("c500"), ["c501"])
        self.assertEqual(g.dependencies("X"), ["c500"])
        self.assertEqual(g.dependencies(self.last), [])

        # 辅助索引未污染：后续合法批次（含跨长链的前向边）仍正常工作
        self.assertIsNone(g.replace_batch({"c0": ["c1", "X"]}))
        self.assertEqual(set(g.dependencies("c0")), {"c1", "X"})
        # 且真正的长链环仍能被识别
        cycle = g.replace_batch({self.last: ["c0"]})
        self.assertIsNotNone(cycle)


class BasicSemanticsTests(unittest.TestCase):
    def test_dependencies_returns_copy(self):
        g = DependencyGraph({"A": ["B"], "B": []})
        d = g.dependencies("A")
        d.append("ZZZ")
        self.assertEqual(g.dependencies("A"), ["B"])

    def test_unknown_node_and_target_rejected(self):
        g = DependencyGraph({"A": ["B"], "B": []})
        with self.assertRaises(KeyError):
            g.dependencies("ZZZ")
        with self.assertRaises(KeyError):
            g.replace_batch({"ZZZ": []})
        with self.assertRaises(KeyError):
            g.replace_batch({"A": ["ZZZ"]})

    def test_batch_size_and_dep_limit(self):
        g = DependencyGraph({i: [] for i in range(200)})
        with self.assertRaises(ValueError):
            g.replace_batch({i: [] for i in range(101)})
        with self.assertRaises(ValueError):
            g.replace_batch({0: list(range(16))})

    def test_self_loop_reports_cycle(self):
        g = DependencyGraph({"A": [], "B": []})
        cycle = g.replace_batch({"A": ["A"]})
        self.assertEqual(cycle, ["A", "A"])
        self.assertEqual(g.dependencies("A"), [])

    def test_cyclic_initial_graph_rejected(self):
        with self.assertRaises(ValueError):
            DependencyGraph({"A": ["B"], "B": ["A"]})

    def test_empty_batch_and_noop_batch(self):
        g = DependencyGraph({"A": ["B"], "B": []})
        self.assertIsNone(g.replace_batch({}))
        self.assertIsNone(g.replace_batch({"A": ["B"]}))  # 原样替换
        self.assertEqual(g.dependencies("A"), ["B"])


class RandomizedDifferentialTests(unittest.TestCase):
    """随机图上与朴素全量重算对照：接受/拒绝一致、环合法、提交内容一致。"""

    def test_against_naive_reference(self):
        rng = random.Random(20261005)
        for trial in range(30):
            n = 12
            nodes = list(range(n))
            # 随机无环图：只允许小编号依赖大编号
            old = {i: [j for j in range(i + 1, n) if rng.random() < 0.2]
                   for i in range(n)}
            g = DependencyGraph(old)
            committed = {i: list(d) for i, d in old.items()}
            for _ in range(40):
                changed = rng.sample(nodes, rng.randint(1, 6))
                changes = {u: rng.sample(nodes, rng.randint(0, 4))
                           for u in changed}
                result = g.replace_batch(changes)
                # 朴素参考：构造候选图并全量判环
                cand = {i: list(d) for i, d in committed.items()}
                for u, deps in changes.items():
                    cand[u] = list(dict.fromkeys(deps))
                edges = {(u, v) for u, ds in cand.items() for v in ds}
                has_cycle = self._naive_has_cycle(nodes, edges)
                if has_cycle:
                    self.assertIsNotNone(result, f"trial {trial} 漏判环")
                    assert_valid_cycle(self, result, edges)
                    # 已提交依赖不变
                    for u in nodes:
                        self.assertEqual(set(g.dependencies(u)),
                                         set(committed[u]))
                else:
                    self.assertIsNone(result, f"trial {trial} 误判环")
                    committed = cand
                    for u in nodes:
                        self.assertEqual(set(g.dependencies(u)),
                                         set(committed[u]))

    @staticmethod
    def _naive_has_cycle(nodes, edges):
        adj = {u: [] for u in nodes}
        for u, v in edges:
            adj[u].append(v)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {u: WHITE for u in nodes}
        for s in nodes:
            if color[s] != WHITE:
                continue
            stack = [(s, iter(adj[s]))]
            color[s] = GRAY
            while stack:
                u, it = stack[-1]
                advanced = False
                for v in it:
                    if color[v] == GRAY:
                        return True
                    if color[v] == WHITE:
                        color[v] = GRAY
                        stack.append((v, iter(adj[v])))
                        advanced = True
                        break
                if not advanced:
                    color[u] = BLACK
                    stack.pop()
        return False


class PerformanceTests(unittest.TestCase):
    """性能：10 万节点长链 + 5000 个单节点批次。

    统计组件真实的边检查次数（edge_checks，含重复检查及辅助入边
    结构中的检查），排除初始化后必须少于 200 万次。
    """

    N = 100_000
    BATCHES = 5_000
    EDGE_CHECK_BUDGET = 2_000_000

    def test_long_chain_forward_batches(self):
        # 初始图：节点 i 依赖节点 i+1 的 10 万节点链
        graph = {i: [i + 1] for i in range(self.N - 1)}
        graph[self.N - 1] = []
        g = DependencyGraph(graph)

        g.edge_checks = 0  # 排除初始化
        rng = random.Random(20261005)  # 固定随机种子，可重复
        half = self.N // 2
        for _ in range(self.BATCHES):
            u = rng.randrange(0, half)           # 变更节点：链的前半段
            v = rng.randrange(half, self.N)      # 附加依赖目标：后半段
            # 每批保留原链边 u->u+1，并添加或替换一条附加依赖 u->v
            result = g.replace_batch({u: [u + 1, v]})
            self.assertIsNone(result)

        self.assertLess(
            g.edge_checks,
            self.EDGE_CHECK_BUDGET,
            f"边检查次数 {g.edge_checks} 超出预算 {self.EDGE_CHECK_BUDGET}",
        )
        print(f"\n[perf] {self.BATCHES} 批次合计边检查次数: {g.edge_checks}")


if __name__ == "__main__":
    unittest.main()
