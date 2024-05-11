from typing import Protocol

from IPython.display import HTML, display


class _DisplayObject(Protocol):
    """Just for type hinting."""

    def _repr_html_(self) -> str: ...


def display_multiple_df(
    *df: _DisplayObject | tuple[str, _DisplayObject],
    **kw,
) -> None:
    # if not all(hasattr(i, "_repr_html_") for i in df):
    #     raise AttributeError("All passed objects must have '_repr_html_' attribute.")

    margin: int = kw.get("margin", 10)

    html = '<div style="display: flex;">'
    for i in df:
        if isinstance(i, tuple):
            html += f"""\
            <div>
                <p align="center">{i[0]}</p>
                <div style="margin-right: {margin}px;">{i[1]._repr_html_()}</div>
            </div>
            """
        else:
            html += f'<div style="margin-right: {margin}px;">{i._repr_html_()}</div>'
    html += "</div>"
    display(HTML(html))
