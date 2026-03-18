#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/repo" >&2
  exit 1
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 1
  }
}

require_clean_tree() {
  git diff --quiet --ignore-submodules --exit-code
  git diff --cached --quiet --ignore-submodules --exit-code
}

write_schemas() {
  cat > "$PLANNER_SCHEMA" <<'JSON'
{
  "type": "object",
  "properties": {
    "status": { "type": "string", "enum": ["PROPOSED", "DONE"] },
    "title": { "type": "string" },
    "goal": { "type": "string" },
    "steps": {
      "type": "array",
      "items": { "type": "string" }
    },
    "acceptance_criteria": {
      "type": "array",
      "items": { "type": "string" }
    },
    "rationale": { "type": "string" }
  },
  "required": [
    "status",
    "title",
    "goal",
    "steps",
    "acceptance_criteria",
    "rationale"
  ],
  "additionalProperties": false
}
JSON

  cat > "$REVIEW_SCHEMA" <<'JSON'
{
  "type": "object",
  "properties": {
    "status": { "type": "string", "enum": ["PASS", "FAIL"] },
    "summary": { "type": "string" },
    "blockers": {
      "type": "array",
      "items": { "type": "string" }
    },
    "warnings": {
      "type": "array",
      "items": { "type": "string" }
    },
    "required_fixes": {
      "type": "array",
      "items": { "type": "string" }
    }
  },
  "required": [
    "status",
    "summary",
    "blockers",
    "warnings",
    "required_fixes"
  ],
  "additionalProperties": false
}
JSON
}

write_planner_prompt() {
  local prompt_file="$1"

  cat > "$prompt_file" <<EOF
You are the planner.

Repository: $REPO_DIR

Task:
- Inspect the current repository state.
- Propose exactly one next refactoring plan that is safe to implement now.
- The plan must be scoped, testable, and realistically finishable in one implementation pass plus review-fix loop.
- Prefer high-value refactors that reduce complexity, duplication, or maintenance risk.
- If no worthwhile next plan remains, return DONE.

Output rules:
- Return JSON only, matching the provided schema exactly.
- If status is PROPOSED:
  - title must be specific and short.
  - goal must be one sentence.
  - steps must contain 2 to 6 concrete implementation steps.
  - acceptance_criteria must contain objective checks.
  - rationale must briefly justify priority.
- If status is DONE:
  - title must be an empty string.
  - goal must be an empty string.
  - steps must be [].
  - acceptance_criteria must be [].
  - rationale must briefly explain why no further safe high-value refactor is available.
EOF
}

write_implement_prompt() {
  local prompt_file="$1"
  local plan_file="$2"

  cat > "$prompt_file" <<EOF
You are the implementer.

Implement the following refactoring plan in the repository.

Rules:
- Make the smallest coherent change set that fully satisfies the plan.
- Preserve behavior unless the plan explicitly says otherwise.
- Update or add tests if changed code requires it.
- Run the smallest relevant validation before finishing.
- Do not commit.
- Do not broaden scope beyond this plan.

Plan JSON:
$(cat "$plan_file")

Final response requirements:
- Briefly list changed files.
- Briefly list validation commands you ran.
- Briefly list any residual risks.
EOF
}

write_fix_prompt() {
  local prompt_file="$1"
  local review_file="$2"

  cat > "$prompt_file" <<EOF
Resume implementation and address exactly the commit-blocking review findings below.

Rules:
- Fix all REQUIRED_FIXES.
- Do not widen scope beyond what is necessary to make the reviewer return PASS.
- Keep the original plan intent intact.
- Run the smallest relevant validation before finishing.
- Do not commit.

Review JSON:
$(cat "$review_file")

Final response requirements:
- Briefly list changed files in this fix pass.
- Briefly list validation commands you ran.
- Briefly list any remaining non-blocking risks.
EOF
}

write_review_prompt() {
  local prompt_file="$1"
  local plan_file="$2"

  cat > "$prompt_file" <<EOF
You are the reviewer.

Review only the current uncommitted git diff and decide whether it is commit-ready right now.

Plan context:
$(cat "$plan_file")

Commit-readiness criteria:
- FAIL if there is a likely bug, behavior regression, broken build/type/test surface, missing necessary tests for changed behavior, or important maintainability regression.
- PASS if the diff is safe to commit now.
- Non-blocking improvements must go into warnings, not blockers.
- required_fixes must contain only items that are necessary for PASS.

Output rules:
- Return JSON only, matching the provided schema exactly.
- blockers and required_fixes should be empty arrays on PASS.
- warnings may be non-empty even on PASS.
EOF
}

run_planner() {
  local prompt_file="$STATE_DIR/plan.prompt.txt"
  local out_file="$STATE_DIR/plan.json"

  write_planner_prompt "$prompt_file"

  (
    cd "$REPO_DIR"
    codex --profile "$PLANNER_PROFILE" exec \
      --output-schema "$PLANNER_SCHEMA" \
      -o "$out_file" \
      "$(cat "$prompt_file")"
  )

  echo "$out_file"
}

start_implementer() {
  local plan_file="$1"
  local prompt_file="$STATE_DIR/implement.prompt.txt"
  local out_file="$STATE_DIR/implement.final.txt"
  local jsonl_file="$STATE_DIR/implement.jsonl"

  write_implement_prompt "$prompt_file" "$plan_file"

  (
    cd "$REPO_DIR"
    codex --profile "$IMPLEMENTER_PROFILE" exec \
      --json \
      -o "$out_file" \
      "$(cat "$prompt_file")" > "$jsonl_file"
  )

  local session_id
  session_id="$(
    jq -r '
      select(.type == "thread.started")
      | .thread_id
    ' "$jsonl_file" | head -n 1
  )"

  [[ -n "$session_id" && "$session_id" != "null" ]] || {
    echo "failed to capture implementer session id: $jsonl_file" >&2
    exit 1
  }

  echo "$session_id"
}

run_reviewer() {
  local review_round="$1"
  local plan_file="$2"
  local prompt_file="$STATE_DIR/review_${review_round}.prompt.txt"
  local out_file="$STATE_DIR/review_${review_round}.json"

  write_review_prompt "$prompt_file" "$plan_file"

  (
    cd "$REPO_DIR"
    codex --profile "$REVIEWER_PROFILE" exec \
      --output-schema "$REVIEW_SCHEMA" \
      -o "$out_file" \
      "$(cat "$prompt_file")"
  )

  echo "$out_file"
}

resume_implementer_with_fixes() {
  local review_round="$1"
  local session_id="$2"
  local review_file="$3"
  local prompt_file="$STATE_DIR/fix_${review_round}.prompt.txt"
  local out_file="$STATE_DIR/fix_${review_round}.final.txt"

  write_fix_prompt "$prompt_file" "$review_file"

  (
    cd "$REPO_DIR"
    codex --profile "$IMPLEMENTER_PROFILE" exec resume "$session_id" \
      -o "$out_file" \
      "$(cat "$prompt_file")"
  )
}

print_review_summary() {
  local review_file="$1"
  jq -r '
    "status: " + .status,
    "summary: " + .summary,
    (
      if (.blockers | length) > 0 then
        "blockers:\n- " + (.blockers | join("\n- "))
      else
        "blockers: none"
      end
    ),
    (
      if (.warnings | length) > 0 then
        "warnings:\n- " + (.warnings | join("\n- "))
      else
        "warnings: none"
      end
    )
  ' "$review_file"
}

require_cmd git
require_cmd jq
require_cmd codex

REPO_DIR="$(cd "$1" && pwd -P)"
[[ -d "$REPO_DIR/.git" ]] || {
  echo "not a git repository: $REPO_DIR" >&2
  exit 1
}

PLANNER_PROFILE="${PLANNER_PROFILE:-planner}"
IMPLEMENTER_PROFILE="${IMPLEMENTER_PROFILE:-implementer}"
REVIEWER_PROFILE="${REVIEWER_PROFILE:-reviewer}"
MAX_REVIEW_FIXES="${MAX_REVIEW_FIXES:-3}"

STATE_DIR="${STATE_DIR:-$REPO_DIR/.codex-orchestrator}"
mkdir -p "$STATE_DIR"

PLANNER_SCHEMA="$STATE_DIR/planner.schema.json"
REVIEW_SCHEMA="$STATE_DIR/review.schema.json"

write_schemas

(
  cd "$REPO_DIR"
  require_clean_tree || {
    echo "working tree must be clean before starting" >&2
    exit 1
  }
)

echo "=== planning ==="
plan_file="$(run_planner)"

plan_status="$(jq -r '.status' "$plan_file")"
[[ "$plan_status" == "PROPOSED" || "$plan_status" == "DONE" ]] || {
  echo "invalid planner status in $plan_file" >&2
  exit 1
}

if [[ "$plan_status" == "DONE" ]]; then
  echo "planner returned DONE"
  exit 0
fi

plan_title="$(jq -r '.title' "$plan_file")"
[[ -n "$plan_title" ]] || {
  echo "planner returned empty title for PROPOSED plan: $plan_file" >&2
  exit 1
}

echo "selected plan: $plan_title"

implementer_session_id="$(start_implementer "$plan_file")"
echo "implementer session: $implementer_session_id"

for (( review_round = 1; review_round <= MAX_REVIEW_FIXES + 1; review_round++ )); do
  echo "=== review round $review_round ==="
  review_file="$(run_reviewer "$review_round" "$plan_file")"
  print_review_summary "$review_file"

  review_status="$(jq -r '.status' "$review_file")"
  [[ "$review_status" == "PASS" || "$review_status" == "FAIL" ]] || {
    echo "invalid reviewer status in $review_file" >&2
    exit 1
  }

  if [[ "$review_status" == "PASS" ]]; then
    echo "review passed"
    echo "stopping without commit"
    echo "working tree is preserved as-is"
    echo "plan title: $plan_title"
    echo "plan file: $plan_file"
    echo "last review file: $review_file"
    exit 0
  fi

  if (( review_round > MAX_REVIEW_FIXES )); then
    echo "review still failing after $MAX_REVIEW_FIXES fix rounds" >&2
    echo "last review file: $review_file" >&2
    exit 1
  fi

  echo "=== fix round $review_round ==="
  resume_implementer_with_fixes "$review_round" "$implementer_session_id" "$review_file"
done

echo "internal error: review loop ended unexpectedly" >&2
exit 1
