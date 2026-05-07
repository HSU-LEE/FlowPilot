from __future__ import annotations
import pytest
import flowpilot as fp

def test_node_forward_reads_inputs_and_writes_output() -> None:
    node = fp.Node(name='sum', inputs=('a', 'b'), outputs=('c',), compute=lambda a, b: a + b, tags=('math',))
    ctx = node.forward(fp.RunContext({'a': 2, 'b': 3}))
    assert ctx.get('c') == 5
    assert node.tags == ('math',)

def test_node_forward_writes_multiple_outputs() -> None:
    node = fp.Node(name='split', inputs=('x',), outputs=('lo', 'hi'), compute=lambda x: (x - 1, x + 1))
    ctx = node.forward(fp.RunContext({'x': 10}))
    assert ctx.get('lo') == 9
    assert ctx.get('hi') == 11

def test_node_validates_multiple_output_shape() -> None:
    node = fp.Node(name='bad', inputs=('x',), outputs=('a', 'b'), compute=lambda x: x)
    with pytest.raises(fp.NodeOutputError):
        node.forward(fp.RunContext({'x': 1}))

def test_node_validates_multiple_output_length() -> None:
    node = fp.Node(name='bad', inputs=('x',), outputs=('a', 'b'), compute=lambda x: (x,))
    with pytest.raises(fp.NodeOutputError):
        node.forward(fp.RunContext({'x': 1}))

def test_pipeline_is_lazy_until_run() -> None:
    calls: list[str] = []
    a = fp.Node('a', inputs=('x',), outputs=('y',), compute=lambda x: calls.append('a') or x + 1)
    b = fp.Node('b', inputs=('y',), outputs=('z',), compute=lambda y: calls.append('b') or y * 2)
    pipeline = a >> b
    assert calls == []
    assert pipeline.names() == ('a', 'b')
    out = pipeline.run(fp.RunContext({'x': 2}))
    assert calls == ['a', 'b']
    assert out.get('z') == 6

def test_pipeline_run_accepts_kwargs() -> None:
    pipeline = fp.Pipeline((fp.Node('a', ('x',), ('y',), lambda x: x + 1),))
    out = pipeline.run(x=4)
    assert out.get('y') == 5

def test_block_as_node_uses_compute_contract() -> None:

    class AddOne(fp.Block):
        name = 'add_one'
        inputs = ('x',)
        outputs = ('y',)

        def forward(self, x):
            return x + 1
    pipeline = AddOne() >> fp.Node('double', ('y',), ('z',), lambda y: y * 2)
    out = pipeline.run(fp.RunContext({'x': 3}))
    assert out.get('z') == 8
