"""DependencyGraph 的自动化测试：两组正确性测试 + 可重复性能测试。"""
import random
import unittest

from dependency_graph import DependencyGraph


def make_chain(n):
    """n 个节点的链：节点 i 依赖节点 i+1。"""
    return DependencyGraph(range(n), {i: [i + 1] for i in range(n - 1)})


def assert_valid_cycle(testcase, cycle, candidate_deps):
    """校验环：首尾相同、中间不重复、每条边真实存在于候选图中。"""
    testcase.assertIsNotNone(cycle)
    testcase.assertGreaterEqual(len(cycle), 2)
    testcase.assertEqual(cycle[0], cycle[-1])
    testcase.assertEqual(len(set(cycle[:-1])), len(cycle) - 1)
    for a, b in zip(cycle, cycle[1:]):
        testcase.assertIn(b, candidate_deps.get(a, ()),
                          msg=f"边 {a}->{b} 不在候选图中")


class TestCorrectnessGroup1(unittest.TestCase):
    """同批撤旧边加反向边；菱形共享依赖不误判。"""

    def test_remove_and_reverse_in_same_batch(self):
        # 旧图只有 A->B；同批清除 A 的依赖并令 B->A。
        # 逐项作用会暂时成环，但最终图无环，必须成功。
        for order in (["A", "B"], ["B", "A"]):  # 交换输入顺序仍成功
            g = DependencyGraph(["A", "B"], {"A": ["B"]})
            changes = {"A": [], "B": ["A"]}
            reordered = {k: changes[k] for k in order}
            self.assertIsNone(g.replace_batch(reordered))
            self.assertEqual(g.dependencies("A"), [])
            self.assertEqual(g.dependencies("B"), ["A"])

    def test_diamond_shared_dependency_not_false_cycle(self):
        # 菱形：A->B, A->C, B->D, C->D，D 被两条路径共享到达。
        # 加入 E->A：从 A 的前向搜索会两次到达 D，不能误判为环。
        g = DependencyGraph(
            ["A", "B", "C", "D", "E"],
            {"A": ["B", "C"], "B": ["D"], "C": ["D"], "E": []},
        )
        self.assertIsNone(g.replace_batch({"E": ["A"]}))
        self.assertEqual(g.dependencies("E"), ["A"])
        # 菱形中真实存在的环仍应被检出：D->A 会闭合 A->B->D->A。
        cyc = g.replace_batch({"D": ["A"]})
        self.assertIsNotNone(cyc)
        assert_valid_cycle(self, cyc, {
            "A": ["B", "C"], "B": ["D"], "C": ["D"], "D": ["A"], "E": ["A"],
        })
        self.assertEqual(g.dependencies("D"), [])  # 拒绝后无残留


class TestCorrectnessGroup2(unittest.TestCase):
    """环必须同时使用两条新边和一段未修改长链；拒绝后状态不受污染。"""

    CHAIN_LEN = 50000  # 长链同时验证深度搜索不溢出调用栈

    def setUp(self):
        # 长链 q0 -> q1 -> ... -> qL，外加 D->A；A 无依赖。
        self.L = self.CHAIN_LEN
        nodes = list(range(self.L + 1)) + ["A", "D"]
        deps = {i: [i + 1] for i in range(self.L)}
        deps["D"] = ["A"]
        self.g = DependencyGraph(nodes, deps)

    def test_cycle_needs_two_new_edges_and_long_chain(self):
        q0, qL = 0, self.L
        # 候选环：A->q0 ->(长链)-> qL ->D ->A。
        # 单独加 A->q0 或 qL->D 都不成环，必须两条新边同时存在。
        changes = {"A": [q0], qL: ["D"]}
        candidate = {i: [i + 1] for i in range(self.L)}
        candidate["D"] = ["A"]
        candidate["A"] = [q0]
        candidate[qL] = ["D"]

        cyc = self.g.replace_batch(changes)
        assert_valid_cycle(self, cyc, candidate)
        # 环确实穿过长链（长度显著大于 2）。
        self.assertGreater(len(cyc), 100)

        # 拒绝后无残留：已提交依赖与旧图一致。
        self.assertEqual(self.g.dependencies("A"), [])
        self.assertEqual(self.g.dependencies(qL), [])
        self.assertEqual(self.g.dependencies("D"), ["A"])

        # 失败后立即提交合法批次：A->q0 单独成立（q0 无法回到 A）。
        self.assertIsNone(self.g.replace_batch({"A": [q0]}))
        self.assertEqual(self.g.dependencies("A"), [q0])

        # 再提交真正成环的批次：qL->D 此时闭合 A->q0->...->qL->D->A。
        cyc2 = self.g.replace_batch({qL: ["D"]})
        assert_valid_cycle(self, cyc2, {
            **{i: [i + 1] for i in range(self.L)},
            "D": ["A"], "A": [q0], qL: ["D"],
        })
        # 辅助索引未被污染：拒绝后 qL 仍无依赖，后续合法批次照常成功。
        self.assertEqual(self.g.dependencies(qL), [])
        self.assertIsNone(self.g.replace_batch({"D": ["A", qL]}))
        self.assertEqual(set(self.g.dependencies("D")), {"A", qL})


class TestPerformance(unittest.TestCase):
    """可重复性能测试：10 万节点链 + 5000 个单节点批次，边检查 < 200 万。"""

    def test_chain_incremental_batches(self):
        n = 100_000
        g = DependencyGraph(range(n), {i: [i + 1] for i in range(n - 1)})
        rng = random.Random(20261004)  # 固定随机种子，测试可重复

        checks_after_init = g.edge_checks
        for _ in range(5000):
            u = rng.randrange(0, n // 2)            # 变更节点：链的前半段
            v = rng.randrange(n // 2, n)            # 附加依赖目标：后半段
            deps = [u + 1, v] if u + 1 < n else [v]  # 保留原链边 + 附加依赖
            self.assertIsNone(g.replace_batch({u: deps}))

        checked = g.edge_checks - checks_after_init
        self.assertLess(checked, 2_000_000,
                        f"验证与索引维护合计检查 {checked} 条边，超出预算")
        print(f"\n[perf] 5000 批次合计边检查次数: {checked}")


if __name__ == "__main__":
    unittest.main()
