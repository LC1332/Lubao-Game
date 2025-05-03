import gradio as gr
import json

# 读取 JSONL 文件
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# 保存标注数据
def save_annotation(data, output_file):
    with open(output_file, 'a') as f:
        f.write(json.dumps(data) + '\n')

# 初始化数据
data = load_data('string/not_match_in_79_data.jsonl')
index = 0

# 更新显示内容
def update_display(index):
    if index < 0 or index >= len(data):
        return None, None
    item = data[index]
    prefix = item.get('prefix', '')
    query = item.get('query', '')
    return prefix, query

# 提交标注
def submit_query(user_query):
    global index
    if 0 <= index < len(data):
        data[index]['query'] = user_query
        save_annotation(data[index], 'string/annotate_query.jsonl')
    return update_display(index)

# 处理按钮事件
def next_item():
    global index
    index += 1
    return update_display(index)

def previous_item():
    global index
    index -= 1
    return update_display(index)

# Gradio 接口
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            prefix_text = gr.TextArea(label="Prefix", interactive=False, lines = 300)
        with gr.Column():
            query_text = gr.Textbox(label="Query")
            submit_btn = gr.Button("提交")
            prev_btn = gr.Button("上一个")
            next_btn = gr.Button("下一个")

    # prefix_text.update(value=update_display(index)[0])
    # query_text.update(value=update_display(index)[1])

    submit_btn.click(submit_query, inputs=query_text, outputs=[prefix_text,query_text])
    next_btn.click(next_item, outputs=[prefix_text,query_text])
    prev_btn.click(previous_item, outputs=[prefix_text,query_text])

# 启动 Gradio 应用
demo.launch()
