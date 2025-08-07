# pip install openai
from openai import OpenAI # openai==1.81.0
import pandas as pd

train_df = pd.read_csv("/data/ephemeral/home/Dialogue_Summarization/data/train.csv")
topic_list = train_df['topic'].value_counts().index.tolist()

def get_topic_prompt(dialogue):
    client = OpenAI(
        api_key="up_5akmKquDPWHEAVLphqnqaSkSruhj8",
        base_url="https://api.upstage.ai/v1"
    )
    # 프롬프트 템플릿 정의
    template = """
    당신은 한국어로 번역된 대화를 읽고, 그 대화의 주요 주제를 9235개 토픽 목록 중에서 가장 관련 있는 항목 1개로 선택해야 합니다.
    대화를 읽고 가장 관련 있는 **하위 주제 하나만** 골라주세요.
    토픽 목록: {topic_list}
    **다른 말 없이 해당 주제 키워드만 출력하세요.**

    [대화]
    {dialogue}
    """

    stream = client.chat.completions.create(
        model="solar-pro2",
        messages=[
            {
                "role": "user",
                "content": template.format(dialogue=dialogue, topic_list=topic_list)
            }
        ],
        reasoning_effort="high", 
        stream=True,
    )
    
    response_content = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            #print(content, end="")
            response_content += content
    
    return response_content

def solar_summary(dialogue, example_dialogue, example_summary):
    
    client = OpenAI(
        api_key="Upstage_API_Key",  ## 여기에 키 입력
        base_url="https://api.upstage.ai/v1"
    )

    # 프롬프트 템플릿 정의
    template = """
    당신은 일상 대화를 요약하는 전문가입니다. 주어진 대화 내용을 분석하여 핵심 정보를 포함하는 간결한 요약문을 작성해 주세요. 요약은 관찰자의 객관적인 관점에서 작성되어야 하며, 각 화자의 의도와 대화의 맥락을 깊이 이해한 후 반영해야 합니다.
    아래의 모든 조건을 **명확하고 직접적인 방식(Direct and Clear Prompts)**으로 충족해야 합니다.
    • 대화의 가장 중요한 정보를 정확하게 전달해야 합니다.
    • 요약문의 길이는 원본 대화 길이의 20% 이내로 제한합니다.
    • 대화 내에 언급된 중요한 명명된 개체(예: 사람이름, 기업명, 특정 장소 등)는 요약문에 반드시 보존되어야 합니다.(한글은 한글로 영어는 영어로 보존)
    • 은어나 약어를 사용하지 않고, 공식적이고 표준적인 한국어를 사용하여 작성해야 합니다.
    • 이 요약은 학교 생활, 직장, 치료, 쇼핑, 여가, 여행 등 광범위한 일상 생활 주제의 대화에 적용됩니다.
    • 출력은 요약문 한 문단만 (다른 설명 없이) 출력하세요.

    [평가지표]
    • 생성된 요약문의 품질은 ROUGE-1, ROUGE-2, ROUGE-L 점수의 평균으로 평가됩니다.

    ### 예시 1 
    [대화 시작]
    {example_dialogue}
    [대화 끝]

    [요약문]
    {example_summary}

    ### 예시 2
    [대화 시작]
    {dialogue}
    [대화 끝]

    위의 대화를 기반으로 한 단락의 요약문을 작성하세요. 다른 설명 없이 요약문만 출력하세요.
    """

    stream = client.chat.completions.create(
        model="solar-pro2",
        messages=[
            {
                "role": "user",
                "content": template.format(dialogue=dialogue, example_dialogue=example_dialogue, example_summary=example_summary)
            }
        ],
        reasoning_effort="high", 
        stream=True,
    )
    
    response_content = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            #print(content, end="")
            response_content += content
    
    return response_content
    # Use with stream=False
    # print(stream.choices[0].message.content)
    #return response_content
    

if __name__ == "__main__":
    
    test_df = pd.read_csv("data/test_topic_with_example.csv")
    test_df['summary'] = test_df.apply(lambda row: solar_summary(row['dialogue'], row['example_dialogue'], row['example_summary']), axis=1)
    test_df.to_csv("test_summary_oneshot_solar.csv", index=False)

    ## test_df 저장 후 파일 확인 후처리 필요할 수 있음. 