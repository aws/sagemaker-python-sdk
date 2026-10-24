"""System prompt and output contract for the rollout agent.

The contract matters for scoring: SOP-Bench's task success rate (TSR) requires every expected
field to be present in the final answer, and a not-applicable field only scores when it is
emitted with a null-like value. So the prompt asks the model to emit ALL fields and to use the
string "NA" for the ones that do not apply.
"""


def build_system_prompt(sop_text: str) -> str:
    # The answer tags deliberately avoid the word "final". gpt-oss confuses a <final_output> /
    # <final_response> tag with its Harmony `final` channel and writes the answer into the
    # reasoning channel, so the endpoint returns HTTP 200 with empty content. Renaming the tags to
    # <answer_json>/<task_result> removed the collision (A/B: empty responses 8.7% -> 0.2%).
    # reward.py maps them back to SOP-Bench's tag names before scoring.
    return (
        "You are an AI assistant that processes requests according to a Standard Operating "
        "Procedure (SOP). Follow each SOP step in order, using the available tools to gather "
        "information and make decisions.\n\n"
        f"<sop>\n{sop_text}\n</sop>\n\n"
        "When ALL SOP steps are complete, stop calling tools and provide your final answer in "
        "<task_result> tags, containing the <answer_json> JSON dictionary as specified in the "
        "SOP. The <answer_json> JSON MUST include EVERY decision field the SOP asks for. For any "
        'field that does not apply to this case, set its value to the string "NA" -- do not omit '
        "the field and do not use null. Example:\n"
        '<task_result><answer_json>{"mechanical_inspection_result": "success", '
        '"component_incident_response": "NA"}</answer_json></task_result>'
    )
