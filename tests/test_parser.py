from __future__ import annotations

from hawker_agent.agent.parser import parse_response
from hawker_agent.models.output import CodeAgentModelOutput


class TestParseResponse:
    def test_python_block(self) -> None:
        text = "分析页面结构\n```python\nnav('https://example.com')\n```"
        result = parse_response(text)
        assert isinstance(result, CodeAgentModelOutput)
        assert result.thought == "分析页面结构"
        assert "nav(" in result.code
        assert result.has_code

    def test_multiple_python_blocks(self) -> None:
        text = (
            "先导航再点击\n"
            "```python\nnav('url1')\n```\n"
            "继续操作\n"
            "```python\nclick('btn')\n```"
        )
        result = parse_response(text)
        assert "nav(" in result.code
        assert "click(" in result.code

    def test_js_named_block(self) -> None:
        text = '分析\n```js api_script\nconsole.log("test")\n```'
        result = parse_response(text)
        assert 'api_script = "' in result.code
        assert result.has_code

    def test_js_and_python_blocks(self) -> None:
        text = (
            "执行计划\n"
            '```js my_js\nalert("hi")\n```\n'
            "```python\nresult = js(my_js)\n```"
        )
        result = parse_response(text)
        # JS named blocks come first
        assert result.code.index("my_js =") < result.code.index("result = js")

    def test_generic_block_fallback(self) -> None:
        text = "思考内容\n```\nsome_code()\n```"
        result = parse_response(text)
        assert result.thought == "思考内容"
        assert "some_code()" in result.code

    def test_no_code_blocks(self) -> None:
        text = "这是纯文本思考，没有代码块"
        result = parse_response(text)
        assert result.thought == text
        assert result.code == ""
        assert not result.has_code

    def test_empty_code_block(self) -> None:
        text = "思考\n```python\n\n```"
        result = parse_response(text)
        # Empty content block is skipped
        assert result.code == ""

    def test_thought_before_first_block(self) -> None:
        text = "第一步：分析页面\n第二步：提取数据\n```python\ndata = extract()\n```"
        result = parse_response(text)
        assert "第一步" in result.thought
        assert "第二步" in result.thought

    def test_empty_input(self) -> None:
        result = parse_response("")
        assert result.thought == ""
        assert result.code == ""
        assert result.is_empty()

    def test_whitespace_only(self) -> None:
        result = parse_response("   \n  ")
        assert result.thought == ""
        assert result.code == ""

    def test_js_without_name_not_captured(self) -> None:
        text = "分析\n```js\nconsole.log('test')\n```"
        result = parse_response(text)
        # JS without a var name is treated as a regular block but not as a named JS block
        # It matches the block_pattern with lang=js but no var_name, so it's neither
        # python nor named-js → skipped from code_parts
        assert result.code == ""

    def test_multiple_generic_blocks(self) -> None:
        text = "思考\n```\nline1()\n```\n中间\n```\nline2()\n```"
        result = parse_response(text)
        assert "line1()" in result.code
        assert "line2()" in result.code

    def test_truncated_unclosed_python_block(self) -> None:
        text = "先分析\n```python\nawait nav('https://example.com')\nobserve('ok')"
        result = parse_response(text)
        assert result.thought == "先分析"
        assert "await nav(" in result.code
        assert "observe('ok')" in result.code
