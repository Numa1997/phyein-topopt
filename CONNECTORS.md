# Connectors and MCP servers for this toolchain, explained at developer level

Written 11 November 2026. Everything here was checked against primary sources or by reading the code
in that session; where something was not checked, it says so.

## 1. What the pieces are

**MCP (Model Context Protocol)** is an open protocol through which an AI assistant calls tools that
live outside it. An **MCP server** is a small program that offers such tools: each tool has a name,
a description and typed arguments (for example `create_object(doc, type, properties)` in FreeCAD).
The assistant decides to call a tool; the server does the work and returns the result.

A **connector** is Claude's name for an MCP server that has been added to your Claude account or app.
There are two kinds, and the difference is where the server program runs:

- **Remote connector**: the server runs on the internet and is reached by an `https://…` URL. It is
  added once to your Claude account and then works on the web, the desktop app and the phone. It can
  only reach what that remote service can reach; it never touches your PC.
- **Local MCP server**: the server runs on your PC, started by Claude Desktop or Claude Code, which
  talk to it over its standard input and output ("stdio"). It can do whatever your user account can
  do on that PC: open FreeCAD, run a solver, read files. That power is the reason for the safety
  rules in section 4.

A cloud session like the one that built this toolchain runs in Anthropic's servers, not on your PC,
so a local MCP server on your PC is not available to it. Local servers serve Claude Desktop or Claude
Code running on your machine.

## 2. Adding a remote connector (by URL)

Per Claude's documentation ("Add a connector by URL", claude.com/docs/connectors/custom):

1. In Claude (web or desktop), open **Settings → Connectors** (in newer versions **Customize →
   Connectors**).
2. Click **Add custom connector** (in some versions: **Add** at the top right, then
   **Add custom connector**).
3. Enter a name and the remote MCP server URL (it usually ends in `/mcp`).
4. Only if the provider says so: open **Advanced settings** and enter the OAuth Client ID and
   Secret they gave you. Otherwise leave them empty.
5. Click **Add**, then **Connect** and sign in to the service if asked.
6. In a chat, switch the connector on or off with the **+** button → **Connectors**.

The Free plan allows one custom connector; Pro and Max allow more. On Team and Enterprise plans an
owner adds it for the organisation. No remote connector is needed for this toolchain: GitHub access
for sessions works through repository attachment (section 5).

## 3. Installing a local MCP server: freecad-mcp (checked, recommended with care)

**What it is.** freecad-mcp lets Claude drive FreeCAD: create parts, set FEM loads, run FreeCAD's
CalculiX analysis, take screenshots. It has two halves: an **addon** that runs inside FreeCAD and
opens a small XML-RPC server on `localhost`, and the **MCP server** (the `freecad-mcp` Python
package) that Claude starts and that forwards Claude's tool calls to that addon.

**Author and status (checked).** Repository github.com/neka-nat/freecad-mcp, MIT licence
("Shirokuma (k tanaka)"), PyPI maintainer e-mail nekanat.stock@gmail.com; about 700 GitHub stars;
version 0.1.30, last commit 7 November 2026.

**Install on Windows, step by step.**

1. Install **FreeCAD 1.1** from freecad.org (Download → Windows installer), run it once, close it.
2. Install **uv** (it provides `uvx`, which runs a Python tool in its own clean environment). Open
   PowerShell (Start → type `PowerShell` → Enter) and run the command from the official uv page,
   docs.astral.sh/uv/getting-started/installation (Windows tab). Close and reopen PowerShell, then
   check: `uvx --version`.
3. Get the addon: `git clone https://github.com/neka-nat/freecad-mcp.git` (or on GitHub: Code →
   Download ZIP, then unzip).
4. Copy the folder `addon\FreeCADMCP` into FreeCAD's module folder, so that this file exists:
   `%APPDATA%\FreeCAD\v1-1\Mod\FreeCADMCP\InitGui.py` (older unversioned installs:
   `%APPDATA%\FreeCAD\Mod\`). Paste `%APPDATA%\FreeCAD` into the Explorer address bar to get there.
5. Start FreeCAD. In the workbench drop-down (toolbar, top), choose **MCP Addon**. In the
   **FreeCAD MCP** toolbar click **Start RPC Server**. Leave **remote access off** (the default).
6. **Claude Desktop**: menu **Settings → Developer → Edit Config**. This opens
   `claude_desktop_config.json`. Put in (pinned to the audited version):
   ```json
   {
     "mcpServers": {
       "freecad": { "command": "uvx", "args": ["freecad-mcp==0.1.30"] }
     }
   }
   ```
   Save, quit Claude Desktop completely (tray icon → Quit) and start it again. The FreeCAD tools now
   appear under the **+** button → connectors / tools list.
   **Claude Code** instead: in a terminal run `claude mcp add freecad -- uvx freecad-mcp==0.1.30`,
   then `claude mcp list` to see it.
7. Test: with FreeCAD open and the RPC server running, ask Claude "create a 100 × 50 × 20 mm box in
   FreeCAD". FreeCAD should show it.

**How to use it safely.** Keep remote access off. Claude Desktop asks before each tool call; read
the request, and never choose "always allow" for `execute_code` or `execute_code_headless`, because
those run arbitrary Python inside FreeCAD with your user rights. Stop the RPC server when you are not
using it.

## 4. Safety checks done before recommending anything

**freecad-mcp 0.1.30: passes.** What was checked: for each of the 29 Python files that run (10 in
the MCP server, 19 in the FreeCAD addon), its imports and every call that touches files, processes
or the network; the network, `exec` and subprocess code read in full; the dependency list; tests,
examples and scripts scanned for the same risky calls. Not every line of every file was read.
Findings:
- No outbound network access anywhere (no `urllib`, `requests` or raw sockets besides its own
  server); it deletes only its own temporary files; the example `.env` holds only a placeholder.
- The addon's server binds to `127.0.0.1` unless you switch on remote access; then it warns when no
  token is set, checks the token with a constant-time comparison (`hmac.compare_digest`) and
  rejects requests coming from a web browser (protection against malicious web pages calling it).
- By design it executes code sent by the assistant (`exec` in FreeCAD; a FreeCAD subprocess for the
  headless mode). That is its purpose, not a hidden behaviour, and the reason for the rules above.
- The package published on PyPI (0.1.30) is byte-identical to the audited source, file by file.

**CAE-Agent-Hub's CalculiX MCP (publisher "Cai-aa"): dropped.** Its code did not show anything
malicious: the solver is started with an argument list and no shell, over stdio only. It fails the
rule that third-party code must come from an identifiable author: commits are by "Cai"
(a Gmail address), the licence names "Thompson Labs", and neither could be tied to a person or an
organisation. It also pins an old MCP SDK (`mcp < 1.8`). Not recommended.

**AnkusDrive and other FreeCAD MCP servers: not checked, so not recommended.** The toolchain does
not need them: its own solver covers voxel topology optimisation, and freecad-mcp covers CAD and
FreeCAD's own FEM.

## 5. GitHub repositories and Claude sessions

- **Cloud sessions** (claude.ai / the Claude app running code in Anthropic's workspace) reach a
  repository only after it is **attached** to the session. Claude requests it; you approve the
  prompt. Reading a public repository works at once. **Pushing** additionally needs the **Claude
  GitHub App** to have access to that repository: on github.com, profile picture → **Settings** →
  **Applications** → **Installed GitHub Apps** → **Claude** → **Configure** → under
  **Repository access** keep **Only select repositories** → **Select repositories** → add the
  repository → **Save**. That gives write access to the repositories listed and nothing else.
- Sessions cannot use GitHub's GraphQL API, so `gh workflow run` fails there; workflows are started
  with the REST endpoint (see the `topopt-toolchain` skill).
- **Claude Code on your PC** uses your own Git login: `git clone https://github.com/<owner>/phyein-topopt`
  and, for the GitHub CLI, `gh auth login` once.
