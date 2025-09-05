import json
import yaml
import os

def convert_json_to_yaml(json_file, yaml_file):
    pipes = []

    # 単一のJSONファイルを処理
    with open(json_file, 'r', encoding='utf-8') as jf:
        json_data = json.load(jf)
        
        # JSONファイルに含まれる各pipeデータを処理
        for i, pipe in enumerate(json_data):
            new_pipe = {
                'closed': pipe.get('closed', False),
                'id': pipe.get('id', i),  # JSONからidを読み取り、なければ連番を使用
                'name': pipe.get('name', ""),
                'segments': []
            }
            
            for checkpoint in pipe.get('path', []):
                new_checkpoint = {
                    'can_overtake': checkpoint.get('can_overtake', False),
                    'point': checkpoint.get('point', []),
                    'radius': checkpoint.get('radius', 0),
                    'shift_from_centre': checkpoint.get('shift_from_centre', 0)
                }
                new_pipe['segments'].append(new_checkpoint)
            
            pipes.append(new_pipe)

    # YAMLファイルに書き込み
    with open(yaml_file, 'w', encoding='utf-8') as yf:
        yaml.dump({'pipes': pipes}, yf, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"{len(pipes)}個のpipeデータを{yaml_file}に変換しました。")

# メイン処理
if __name__ == "__main__":
    json_file = '/home/y-kobayashi/maps/default/pipes/input.json'  # 入力JSONファイル
    yaml_file = '/home/y-kobayashi/maps/default/pipes/output.yaml'  # 出力するYAMLファイル名
    
    convert_json_to_yaml(json_file, yaml_file)
    print(f"{json_file}から{yaml_file}へ変換しました。")