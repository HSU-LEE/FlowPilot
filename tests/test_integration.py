from __future__ import annotations
import flowpilot as fp
from flowpilot.ops import distance

def test_observe_predict_evaluate_pipeline() -> None:
    observe = fp.Node(name='observe', inputs=('raw',), outputs=('self.position', 'target.position', 'target.velocity'), compute=lambda raw: (raw['me'], raw['target']['position'], raw['target']['velocity']))
    predict = fp.Node(name='predict', inputs=('target.position', 'target.velocity', 'runtime.dt'), outputs=('target.predicted',), compute=lambda p, v, dt: [p[i] + v[i] * dt for i in range(len(p))])
    evaluate = fp.Node(name='evaluate', inputs=('self.position', 'target.predicted'), outputs=('score.distance',), compute=distance.euclidean)
    pipeline = observe >> predict >> evaluate
    loop = fp.TickLoop(pipeline, dt=0.5)
    ctx = loop.run(fp.RunContext({'raw': {'me': [0.0, 0.0], 'target': {'position': [3.0, 4.0], 'velocity': [2.0, 0.0]}}}))
    assert ctx.get('target.predicted') == [4.0, 4.0]
    assert ctx.get('score.distance') == 32 ** 0.5

def test_pipeline_direct_run_is_enough_for_unit_tests() -> None:
    pipeline = fp.Node('observe', ('raw',), ('x',), lambda raw: raw['x']) >> fp.Node('double', ('x',), ('y',), lambda x: x * 2)
    ctx = pipeline.run(raw={'x': 21})
    assert ctx.get('y') == 42
