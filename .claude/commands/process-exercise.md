# Process Exercise Sheet

You are scaffolding a new HTN (Hands-on Tensor Networks) exercise sheet into the Qritical.jl documentation system. The user has uploaded a PDF exercise sheet. Your job is to:

1. Read the PDF the user has provided (path given as argument, or ask if missing)
2. Parse the exercise number and all numbered questions from the PDF
3. Create the matching directory and task files in `docs/src/exercises/`
4. Create a GitHub milestone and one issue per question
5. Update `docs/make.jl` to include the new exercise

---

## Step 1 — Read the PDF

Use the `Read` tool to read the PDF at the path the user provided. Extract:
- The exercise number (e.g. "Ex 5" → number `5`)
- The due date printed on the PDF (format dd.mm.yyyy — convert to ISO 8601 for GitHub: yyyy-mm-ddT00:00:00Z)
- For each numbered question: the question number, the title (short descriptive name), and the full question body text (including any sub-questions marked A), B), etc.)

---

## Step 2 — Create the directory

The exercise directory is:
```
docs/src/exercises/<NN>/
```
where `<NN>` is the exercise number zero-padded to 2 digits (e.g. exercise 5 → `05`).

Check if it already exists with `ls`. If it does, confirm with the user before overwriting anything.

---

## Step 3 — Create task_N.jl files

Create one `task_N.jl` file per question (N = 1, 2, 3, ...).

**Exact format** (Literate.jl compatible):

```julia
# # Task X.N — Title

# !!! question "Task X.N — Title"
#     Full question text here.
#     Continue on next line if needed.
#
#     Sub-questions go here indented the same way.
#
#     !!! subquestion
#         **A)** Sub-question A text
#
#     !!! subquestion
#         **B)** Sub-question B text

using Qritical: QriticalUtils
#--

DATA_ROOT = normpath(joinpath(@__FILE__, ".."))
#--
```

Rules:
- `X` = exercise number, `N` = question number within the exercise
- The title after the `—` dash is a short descriptive name you derive from the question text
- All question body lines are prefixed with `#     ` (4 spaces after `#`)
- LaTeX inline math uses double backticks: `` ``\sigma_z`` ``
- LaTeX display math uses the `math` code block inside admonitions
- If a question has sub-parts (A, B, C...), add them as `!!! subquestion` blocks after the main question text
- Keep the boilerplate `using Qritical: QriticalUtils` and `DATA_ROOT` block at the bottom — omit data file path lines unless the question text explicitly references loading data files

---

## Step 3b — Create GitHub milestone and issues

**Repo:** `knottyanyon/Qritical.jl`

### Milestone

The milestone title is `Week-<NN>` where `<NN>` is the exercise number zero-padded to 2 digits (e.g. exercise 5 → `Week-05`).

First check if it already exists:
```bash
gh api repos/knottyanyon/Qritical.jl/milestones --jq '.[] | select(.title == "Week-<NN>") | .number'
```

If it does not exist, create it using the due date extracted from the PDF:
```bash
gh api repos/knottyanyon/Qritical.jl/milestones \
  --method POST \
  --field title="Week-<NN>" \
  --field due_on="<ISO8601 due date>"
```

Capture the milestone number from the response (field `.number`) — you need it for the issues.

If it already exists, use the number returned by the check query.

### Issues

Create one GitHub issue per question. Issues have:
- **Title:** `X.N Short Title` — the same short title used in `make.jl` (2–4 words, no quotes)
- **Body:** empty (leave blank, matching the existing issue convention)
- **Milestone:** the milestone number obtained above

Create each issue with:
```bash
gh api repos/knottyanyon/Qritical.jl/issues \
  --method POST \
  --field title="X.N Short Title" \
  --field body="" \
  --field milestone=<milestone_number>
```

Create issues in order (question 1 first, then 2, etc.). After each creation, print the issue URL so the user can see it.

---

## Step 4 — Update docs/make.jl

Read the current `docs/make.jl`. Find the `pages=[...]` section and add a new `"Exercise XX"` block following the existing pattern.

**Pattern to match:**
```julia
"Exercise 04" => [
    "4.2 MPS Overlap" => "exercises/04/task_1.md",
    "4.3 Observables" => "exercises/04/task_2.md",
    "4.4 Adding MPS" => "exercises/04/task_3.md",
],
```

**What to add** (example for exercise 5 with 4 questions):
```julia
"Exercise 05" => [
    "5.1 Short Title 1" => "exercises/05/task_1.md",
    "5.2 Short Title 2" => "exercises/05/task_2.md",
    "5.3 Short Title 3" => "exercises/05/task_3.md",
    "5.4 Short Title 4" => "exercises/05/task_4.md",
],
```

Insert the new block **inside** the `"Exercises" => [...]` array, after the last existing exercise entry. Use `Edit` to make the change precisely — find the closing `],` of the last exercise block and insert before the outer `]`.

The short title in `make.jl` should be a 2–4 word summary of the question topic (not the full question text).

---

## Step 5 — Report

After finishing, print a summary:
- List the task files created
- List the GitHub issues created (with URLs)
- Show the milestone that was created or reused
- Show the make.jl region that was changed
- Mention that the `.md` files will be auto-generated by Literate.jl when `make.jl` runs (do not create them manually)
