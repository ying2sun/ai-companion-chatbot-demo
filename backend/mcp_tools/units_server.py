"""
backend/mcp_tools/units_server.py
------------------------------------
A local MCP server exposing unit conversion as a tool the language
model can call directly while generating a reply, rather than only
catching a measurement after the fact by scanning finished text
(suggestions/units.py, the earlier approach). Same conversion math
either way, imported from there, not duplicated, this file is a thin
protocol wrapper around it.

Why this is a real capability gain, not just a different way to do the
same thing: the reply-text scanner only ever fires if the model already
wrote a measurement into its answer on its own. A direct question like
"how many pounds is 90 kg" doesn't guarantee that, the model could just
do the arithmetic itself, which is exactly the unreliable path this
feature exists to avoid. As a callable tool, the model can reach for it
proactively instead of guessing.

Run standalone to test manually:
    python mcp_tools/units_server.py
Normal operation launches this as a subprocess over stdio, from
llm/client.py, nothing else runs it directly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastmcp import FastMCP

from suggestions.units import convert as _convert

mcp = FastMCP(name="units")


@mcp.tool
def convert_units(value: float, from_unit: str) -> dict:
    """
    Convert a measurement to its natural counterpart unit.

    Supported from_unit values: C, F, kg, lb, km, miles, cm.
    Temperature converts to the other temperature scale. Weight and
    distance convert to their common alternate unit. Height (cm)
    converts to feet and inches.

    Returns a dict with from_value, from_unit, and result on success,
    or an "error" key if from_unit isn't recognized.
    """
    return _convert(value, from_unit)


if __name__ == "__main__":
    mcp.run()
