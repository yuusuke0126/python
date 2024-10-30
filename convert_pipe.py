import yaml
import json

def convert_yaml_to_json(yaml_file, json_file):
    # YAMLファイルを読み込む
    with open(yaml_file, 'r', encoding='utf-8') as yf:
        data = yaml.safe_load(yf)
    
    new_pipe = {}
    new_pipe['id'] = 0
    new_pipe['sector_id'] = 1
    new_pipe['location_id'] = 1
    new_pipe['name'] = "Pipe 0"
    new_pipe['closed'] = bool(data.get('closed', False))
    new_pipe['path'] = []
    
    for j, checkpoint in enumerate(data.get('segments', [])):
        checkpoint['id'] = j
        checkpoint['name'] = ""
        new_order = ["can_overtake", "id", "name", "point", "radius", "shift_from_centre"]
        checkpoint = {key: checkpoint[key] for key in new_order if key in checkpoint}
        new_pipe['path'].append(checkpoint)
    
    # JSONファイルに書き込み
    with open(json_file, 'w', encoding='utf-8') as jf:
        json.dump([new_pipe], jf, ensure_ascii=False, indent=2)
    
    print(f"{json_file}に変換しました。")

# メイン処理
if __name__ == "__main__":
    yaml_file = 'input.yaml'  # 変換元のYAMLファイル名
    json_file = 'output.json'  # 変換先のJSONファイル名
    
    convert_yaml_to_json(yaml_file, json_file)
    print(f"{yaml_file} から {json_file} へ変換しました。")