from pathlib import Path

import gradio as gr

from util import MuleDetector, line_notify

MARKDOWN = """
# 金融業ATM防詐守衛告警系統
## 操作流程
1. 上傳疑似車手圖片
2. 點擊「開始分析」
3. 等待系統分析
4. 獲取分析結果，若圖片中疑似出現車手，則系統會發出警告。並且透過LINE通知銀行專員。

## 什麼是「車手」? 
1. 詐騙集團騙取被害人將帳戶裡的金錢提領出來後會派人取款，這個人就是所謂的「車手」
2. 通常車手並不了解實際的狀況，只是有人叫他去做什麼，他就去做什麼，然後在過程中獲取一定額度的報酬。

## 「車手」的特徵
### 重點整理
1. **大多為年輕人**
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
"""


SYSTEM_PROMPT = """
# CONTEXT #
## 什麼是「詐騙車手」? 
1. 詐騙集團騙取被害人將帳戶裡的金錢提領出來後，詐騙集團會派人取款，而這個人就是所謂的「詐騙車手」
2. 通常車手並不了解實際的狀況，只是有人叫他去做什麼，他就去做什麼，然後在過程中獲取一定額度的報酬。

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
透過用戶上傳的圖片進行情緒分析，然後判斷圖片中出現的人是否為詐騙車手

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
量化後的機率與結果(機率取小數點後3位)
"""
THRESHOLD = 0.5


def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=False)


def back_to_files():
    return gr.update(visible=False), gr.update(visible=True), None, None, None


def replay(state: bool, warning_audio: gr.Audio) -> tuple[bool, gr.Audio]:
    if state:
        return state, warning_audio
    return state, warning_audio.stop()


def warning(image_paths: list[Path]):
    warning_audio = gr.update(
        "warning_audio",
        value="warning.mp3",
        visible=False,
        autoplay=True,
    )
    line_notify("警告，目前有車手正在操作ATM", image_paths)
    alert = gr.update("alert", value="alert.mp3", visible=False)
    return warning_audio, alert


def detect(image_paths: list[str]) -> tuple[str, dict, gr.Audio, gr.Audio]:
    image_paths = [Path(image_path) for image_path in image_paths]
    detector = MuleDetector(SYSTEM_PROMPT)
    results, usage = detector.start(
        image_paths,
        "請幫我分析圖片中的人物",
        response_format={"type": "json_object"},
    )
    anals = []
    mules_img = []
    for i, result in enumerate(results.result):
        if result.prob >= THRESHOLD or result.flagged:
            gr.Warning(f"第{i+1}張圖片({image_paths[i].name})中的人物被標記為詐騙車手")
            mules_img.append(image_paths[i])

        anals.append(f"第{i+1}張圖片({image_paths[i].name})的情緒分析結果: {result.analysis}")

    if mules_img:
        warning_audio, alert = warning(mules_img)
    else:
        warning_audio = gr.update("warning_audio", visible=False)
        alert = gr.update("alert", visible=False)

    usage_text = f"""
本次分析共消耗{usage.total_tokens}個Token
(輸入: {usage.prompt_tokens}個Token, 輸出: {usage.completion_tokens}個Token)
    """
    return "\n\n".join(anals), usage_text, warning_audio, alert


with gr.Blocks(title="金融業ATM防詐守衛告警系統") as demo:
    # Build the UI
    gr.Markdown(MARKDOWN)

    with gr.Column():
        image_input = gr.Files(
            label="上傳疑似車手圖片",
            elem_id="image_input",
            type="filepath",
        )
        uploaded_files = gr.Gallery(
            label="圖片預覽",
            elem_id="gallery",
            type="image",
            visible=False,
            columns=4,
            object_fit="contain",
        )

    with gr.Row(visible=True) as clear_button:
        submit_button = gr.Button(value="開始分析", elem_id="submit_button")
        remove_image = gr.ClearButton(
            value="Clear",
            components=image_input,
        )

    with gr.Column():
        output_text = gr.Textbox(label="分析結果", elem_id="output_text")
        usage = gr.Textbox(label="Token消耗", elem_id="usage", interactive=False)
        warning_audio = gr.Audio(elem_id="warning_audio", visible=False, type="filepath")
        alert = gr.Audio(
            elem_id="alert",
            visible=False,
            autoplay=False,
            type="filepath",
        )
    # Set the callbacks
    submit_button.click(
        detect,
        inputs=[image_input],
        outputs=[output_text, usage, warning_audio, alert],
    )
    warning_audio.change(lambda: gr.update(autoplay=True), outputs=[alert]).then(
        lambda: gr.update(autoplay=True),
        outputs=[warning_audio],
    )
    image_input.upload(
        fn=swap_to_gallery, inputs=image_input, outputs=[uploaded_files, image_input]
    ).then(
        lambda: gr.update(value=None),
        outputs=warning_audio,
    )
    remove_image.click(
        fn=back_to_files, outputs=[uploaded_files, image_input, warning_audio, output_text, usage]
    ).then(
        lambda: gr.update(value=None),
        outputs=[alert],
    )


if __name__ == "__main__":
    demo.launch()
