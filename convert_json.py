import json
import yaml
import os

def convert_json_to_yaml(json_directory, yaml_file):
    pipes = []

    # JSONディレクトリ内のすべてのJSONファイルを処理
    for i, filename in enumerate(sorted(os.listdir(json_directory))):
        if filename.endswith('.json'):
            with open(os.path.join(json_directory, filename), 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)
                
                # JSONファイルには1つのpipeデータが含まれていると仮定
                pipe = json_data[0]
                
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
    json_directory = './'  # JSONファイルが格納されているディレクトリ
    yaml_file = 'output.yaml'  # 出力するYAMLファイル名
    
    convert_json_to_yaml(json_directory, yaml_file)
    print(f"{json_directory}ディレクトリ内のJSONファイルから{yaml_file}へ変換しました。")
