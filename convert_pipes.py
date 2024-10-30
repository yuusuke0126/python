import yaml
import json

# YAMLファイルからJSONファイルへの変換関数
def convert_yaml_to_json(yaml_file):
    # YAMLファイルを読み込む
    with open(yaml_file, 'r', encoding='utf-8') as yf:
        data = yaml.safe_load(yf)
    
    for i, pipe in enumerate(data['pipes']):
        new_pipe = {
            'id': i,
            'sector_id': 1,
            'location_id': 1,
            'name': f"Pipe {i}",
            'closed': bool(pipe['closed']),
            'path': []
        }
        
        for j, checkpoint in enumerate(pipe['segments']):
            new_checkpoint = {
                'can_overtake': checkpoint.get('can_overtake', False),
                'id': j,
                'name': "",
                'point': checkpoint.get('point', []),
                'radius': checkpoint.get('radius', 0),
                'shift_from_centre': checkpoint.get('shift_from_centre', 0)
            }
            new_pipe['path'].append(new_checkpoint)
        
        json_file = f"Pipe {i}.json"
        with open(json_file, 'w', encoding='utf-8') as jf:
            json.dump([new_pipe], jf, ensure_ascii=False, indent=2)
        
        print(f"{json_file}を作成しました。")

# メイン処理
if __name__ == "__main__":
    yaml_file = '/home/y-kobayashi/maps/honda_kumamoto/pipes/input.yaml'  # 変換元のYAMLファイル名
    convert_yaml_to_json(yaml_file)
    print(f"{yaml_file}からJSONファイルへの変換が完了しました。")
