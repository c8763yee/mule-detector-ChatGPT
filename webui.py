from asyncio import run
from pathlib import Path

import edge_tts
import gradio as gr

from chat import MuleDetector

MARKDOWN = """
# 金融業ATM防詐守衛告警系統
## 步驟
1. 上傳疑似車手圖片
2. 點擊「開始分析」
3. 等待系統分析
4. 獲取分析結果

若圖片中出現的人被標記為詐騙車手，則系統會發出警告。
"""


SYSTEM_PROMPT = """
# CONTEXT #
## 什麼是「詐騙車手」? 
1. 詐騙集團騙取被害人將帳戶裡的金錢提領出來後，詐騙集團會派人出面向被害人取款，這個取款的人就是所謂的「詐騙車手」
2. 通常車手並不了解實際的狀況，只是有人叫他去做什麼，他就去做什麼，然後在做某一件事的過程中獲取一定額度的報酬。

## 詐騙車手的特徵
### 重點整理
1. **提款車手大多為年輕人**
   - 年輕人在ATM提款時，常因緊張而四處張望，顯得不自在。

2. **掩飾身份**
   - 通常戴上口罩或安全帽，有時甚至全身裝扮都極為考究，進而混淆視聽。

3. **頻繁操作手機**
   - 在提款過程中，經常查看手機，或是接收來自同夥的指令訊息。

4. **使用多張提款卡**
   - 為了提領更多的非法所得，車手通常持有多張提款卡，或反覆使用同一張卡多次提款。

5. **長時間佔用ATM**
   - 因多次操作而長時間佔用ATM機，導致其他等待提款的民眾不耐煩排長隊。

6. **頻繁更換提款地點**
   - 在短時間內多次進出同一家便利店，或在不同的銀行、超市、商場尋找ATM機進行提款。

7. **其他**
   - 講電話聽從指示更改目的地
   - 服裝與年紀不匹配
   - 配戴投資理財公司工作證
   - 頻繁使用提款機或找年長者面交大額現金

# OBJECTIVE #
透過用戶上傳的圖片進行情緒分析，然後判斷圖片中出現的人是否為詐騙車手。最後量化結果(從0到1, 1代表100%是詐騙車手)

# TONE #
僅使用無縮排與換行的JSON格式

# AUDIENCE #
對於提到的人是不是詐騙車手感到疑惑的人

# STYLE #
{"result":[{"prob": <float>, "flagged": <bool>, "analysis": <str>},...]}
prob: 該人是詐騙車手的機率(小數點後3位)
flagged: 是否被標記為詐騙車手
analysis: 情緒分析結果

# RESPONSE #
量化後的結果(取小數點後6位)
"""
THRESHOLD = 0.5


async def text2voice(
    text: str,
    voice: str,
    filepath: Path,
    *,
    rate: str,
    volume: str,
    pitch: str,
    proxy: str | None = None,
    receive_timeout: int = 60,
) -> str:
    """Convert text to voice using edge_tts

    Args:
        text (str): The text to convert to voice
        voice (str, optional): TTS voices. Defaults to settings.VOICE.
        rate (str, optional): voice speed (format: "(+|-)x%"). Defaults to -15%.
        volume (str, optional): voice volume (format: "(+|-)x%"). Defaults to +0%.
        pitch (str, optional): _description_. Defaults to +0Hz.
        proxy (str | None, optional): _description_. Defaults to None.
        filepath (str, optional): _description_. Defaults to "media/dummy.mp3".

    Returns:
        str: The filepath of the generated audio
    """
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        volume=volume,
        pitch=pitch,
        proxy=proxy,
        receive_timeout=receive_timeout,
    )
    await communicate.save(filepath)
    return filepath.absolute()


def tts(text: str) -> Path:
    return run(
        text2voice(
            text,
            "zh-TW-HsiaoChenNeural",
            rate="+0%",
            volume="+30%",
            pitch="+0Hz",
            filepath=Path("warning.mp3"),
        )
    )


def detect(image_paths: list[str]) -> tuple[str, dict, gr.Audio]:
    image_paths = [Path(image_path) for image_path in image_paths]
    detector = MuleDetector(SYSTEM_PROMPT)
    results, usage = detector.start(
        image_paths,
        "請幫我分析圖片中的人物",
        response_format={"type": "json_object"},
    )
    flagged = False
    anals = []
    for i, result in enumerate(results.result):
        if result.prob >= THRESHOLD or result.flagged:
            gr.Warning(f"第{i+1}張圖片({image_paths[i].name})中的人物被標記為詐騙車手")
            flagged = True

        anals.append(f"圖片{i+1}的情緒分析結果: {result.analysis}")

    if flagged:
        warning_audio = gr.update(
            "warning_audio",
            value=str(tts("警告，目前有車手在操作ATM")),
            visible=False,
            autoplay=True,
        )
    return "\n\n".join(anals), usage, warning_audio


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)

    with gr.Row():
        image_input = gr.File(
            label="上傳圖片",
            type="filepath",
            elem_id="image_input",
            scale=2,
            file_count="multiple",
        )
    submit_button = gr.Button(value="開始分析", elem_id="submit_button")
    output_text = gr.Textbox(label="分析結果", elem_id="output_text")
    usage = gr.Textbox(label="Token消耗", elem_id="usage", interactive=False)
    warning_audio = gr.Audio(elem_id="warning_audio", visible=False, type="filepath")
    submit_button.click(
        detect,
        inputs=[image_input],
        outputs=[output_text, usage, warning_audio],
    )
    image_input.upload(
        lambda: gr.update(value=None),
        outputs=warning_audio,
    )
    warning_audio.change(lambda: gr.update(autoplay=True), outputs=warning_audio)


if __name__ == "__main__":
    demo.launch()
