import json
import yaml
import os
import tkinter as tk
from tkinter import filedialog

def get_initial_directory():
    # ホームディレクトリのパスを取得
    home = os.path.expanduser('~')
    
    # 優先順位に従ってディレクトリをチェック
    default_paths = [
        os.path.join(home, 'maps', 'default'),
        os.path.join(home, 'maps'),
        os.getcwd()  # カレントディレクトリ
    ]
    
    # 最初に見つかった存在するディレクトリを返す
    for path in default_paths:
        if os.path.exists(path):
            return path
    
    return os.getcwd()  # 全て存在しない場合はカレントディレクトリを返す

def select_files():
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを非表示

    # 初期ディレクトリを取得
    initial_dir = get_initial_directory()

    # 入力JSONファイルの選択
    json_file = filedialog.askopenfilename(
        title='Select JSON file',
        initialdir=initial_dir,
        filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
    )
    
    if not json_file:
        print("JSONファイルが選択されませんでした。")
        return None, None

    # 出力YAMLファイルの保存場所の選択（JSONファイルと同じディレクトリを初期表示）
    yaml_initial_dir = os.path.dirname(json_file) if json_file else initial_dir
    yaml_file = filedialog.asksaveasfilename(
        title='Save YAML file',
        initialdir=yaml_initial_dir,
        defaultextension='.yaml',
        filetypes=[('YAML files', '*.yaml'), ('All files', '*.*')]
    )
    
    if not yaml_file:
        print("YAMLファイルの保存場所が選択されませんでした。")
        return None, None

    return json_file, yaml_file

def convert_json_to_yaml(json_file, yaml_file):
    pipes = []

    # 単一のJSONファイルを処理
    with open(json_file, 'r', encoding='utf-8') as jf:
        json_data = json.load(jf)
        
        # JSONファイルに含まれる各pipeデータを処理
        for i, pipe in enumerate(json_data):
            new_pipe = {
                'closed': 1 if pipe.get('closed', False) else 0,
                'id': pipe.get('id', i),  # JSONからidを読み取り、なければ連番を使用
                'name': pipe.get('name', ''),
                'segments': []
            }
            
            for checkpoint in pipe.get('path', []):
                # 順番を指定して新しいチェックポイントを作成
                point = checkpoint.get('point', {})
                new_checkpoint = {
                    'point': {
                        'x': float(point.get('x', 0)),  # 明示的にfloat表記
                        'y': float(point.get('y', 0)),
                        'z': float(point.get('z', 0))  # 明示的に0.0を使用
                    },
                    'radius': float(checkpoint.get('radius', 0)),
                    'shift_from_centre': float(checkpoint.get('shift_from_centre', 0)),
                    'can_overtake': checkpoint.get('can_overtake', 0)
                }
                new_pipe['segments'].append(new_checkpoint)
            
            pipes.append(new_pipe)

    # YAMLファイルに書き込み
    class literal_str(str): pass
    def literal_presenter(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")
    yaml.add_representer(literal_str, literal_presenter)

    # nameフィールドの値をliteral_str型に変換
    for pipe in pipes:
        pipe['name'] = literal_str(pipe['name'])

    with open(yaml_file, 'w', encoding='utf-8') as yf:
        yaml.dump({'pipes': pipes}, yf, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"{len(pipes)}個のpipeデータを{yaml_file}に変換しました。")

# メイン処理
if __name__ == "__main__":
    json_file, yaml_file = select_files()
    
    if json_file and yaml_file:
        convert_json_to_yaml(json_file, yaml_file)
        print(f"{json_file}から{yaml_file}へ変換しました。")