# Study 4 error analysis (implicit condition, Qwen2.5-7B vs Llama-3.1-8B)

Agreement on the 800 implicit items (Llama, Qwen): both right 353, Llama only 235, Qwen only 57, both wrong 155.
I read 10 random "Llama only, embedding also wrong" items and 10 "both wrong" items (`seed 0`).

## What the cases look like

**Llama right, Qwen and embedding wrong: a concrete cue buried inside another task.**
The user asks to edit, translate or polish a text, and the preference appears as a detail inside that text.
- Curing salmon with dill and juniper while complaining about work calls → *loves Scandinavian cuisine*
- "Do you think the Twice reference works here?" in an email rewrite → *likes K-pop / J-pop*
- On the ladder stringing lights, setting the fog machine, in a note to a colleague → *elaborate Halloween displays*

**Both wrong: the cue is abstract or only adjacent to the preference.**
- "Why did early monastic communities emphasise shared silence?" → *respects quiet time in shared spaces*
- "Which breathing sequences calm the mind?" → *practises yoga*
- "Why does early blues evoke narrative intimacy?" → *prefers jazz and blues to Indian classical* (the contrast part is not in the context at all)
- "Metrics to compare two focus tools" → *tries new productivity apps*

Some items are hardly answerable: the snippet only asks for a rewrite and the preference is weak or absent. This sets a ceiling below 1 for the implicit condition.

## Quantitative splits (implicit accuracy)

| split | n | Qwen | Llama | embedding |
|---|---|---|---|---|
| preference inside an edit/translate task | 405 | 0.50 | 0.73 | 0.47 |
| direct question / chat about the topic | 395 | 0.53 | 0.74 | 0.62 |
| 1 user turn | 493 | 0.51 | 0.70 | 0.54 |
| 2+ user turns | 307 | 0.52 | 0.79 | 0.55 |

- The embedding drops by 0.15 when the cue is wrapped in another task; both LLMs barely change. Llama's margin over the embedding is largest exactly there (+0.25).
- On direct questions the embedding beats Qwen (0.62 vs 0.53): Qwen does not even use cues that are lexically obvious. Qwen's problem is not "too subtle", it is that conversation content is not treated as information about the user.
- More user turns help Llama (+0.09), not Qwen.
- Correct replies that name the preference ("Since you enjoy…", 278 items) are only slightly easier (+0.04), so the result is not just string matching.

## Stereotypical vs anti-stereotypical, controlled

The raw +0.09 (stereotypical > anti) shrinks after subtracting each item's no-context accuracy:
lift over none, stereotypical − anti = Qwen +0.05 [−0.05, 0.13], Llama +0.07 [−0.03, 0.16] (persona bootstrap). **Not significant.**
But on items the embedding gets wrong (no surface cue), the gap is large: Qwen 0.56 vs 0.39, Llama 0.77 vs 0.63 (stereotypical n=77).
Reading: when the context gives no surface cue, models lean on what is typical. Suggestive only, small n.
