players = {
    'player1': {
        'name': 'プレーヤー1',
        'levels': [90, 72, 75, 75, 75, 73, 72, 74, 65, 1, 1, 56, 67, 60, 71, 37],
        'jobs': ['バトマス', '賢者', 'レンジャ', '魔法戦士', 'パラディ', 'スパスタ', '海賊', 
                'まもマス', 'ゴドハン', '大魔導師', '大神官', 'ニンジャ', '魔剣士', '守り人', 
                'ドラゴン', '天地雷鳴']
    },
    'player2': {
        'name': 'プレーヤー2',
        'levels': [72, 72, 70, 78, 83, 72, 72, 72, 64, 69, 51, 49, 70, 42, 53, 47],
        'jobs': ['バトマス', '賢者', 'レンジャ', '魔法戦士', 'パラディ', 'スパスタ', '海賊', 
                'まもマス', 'ゴドハン', '大魔導師', '大神官', 'ニンジャ', '魔剣士', '守り人', 
                'ドラゴン', '天地雷鳴']
    },
    'player3': {
        'name': 'プレーヤー3',
        'levels': [75, 72, 89, 74, 72, 72, 80, 72, 65, 58, 1, 68, 72, 19, 56, 37],
        'jobs': ['バトマス', '賢者', 'レンジャ', '魔法戦士', 'パラディ', 'スパスタ', '海賊', 
                'まもマス', 'ゴドハン', '大魔導師', '大神官', 'ニンジャ', '魔剣士', '守り人', 
                'ドラゴン', '天地雷鳴']
    },
    'player4': {
        'name': 'プレーヤー4',
        'levels': [75, 90, 75, 75, 75, 81, 75, 75, 27, 61, 66, 51, 71, 35, 63, 37],
        'jobs': ['バトマス', '賢者', 'レンジャ', '魔法戦士', 'パラディ', 'スパスタ', '海賊', 
                'まもマス', 'ゴドハン', '大魔導師', '大神官', 'ニンジャ', '魔剣士', '守り人', 
                'ドラゴン', '天地雷鳴']
    }
}

def filter_jobs_and_levels(players):
    filtered_players = {}
    
    for player_id, player_data in players.items():
        filtered_levels = []
        filtered_jobs = []
        
        # プレーヤー2の特別処理
        if player_id == 'player2':
            for job, level in zip(player_data['jobs'], player_data['levels']):
                if job == '天地雷鳴':
                    filtered_jobs.append(job)
                    filtered_levels.append(level)
        else:
            # 他のプレーヤーの処理
            for job, level in zip(player_data['jobs'], player_data['levels']):
                # 上級職（インデックス0-7）の処理
                if player_data['jobs'].index(job) <= 7:
                    if level < 75:
                        filtered_jobs.append(job)
                        filtered_levels.append(level)
                # 特級職（インデックス8-15）の処理
                else:
                    if level < 70:
                        filtered_jobs.append(job)
                        filtered_levels.append(level)
        
        filtered_players[player_id] = {
            'name': player_data['name'],
            'levels': filtered_levels,
            'jobs': filtered_jobs
        }
    
    return filtered_players

# フィルタリングを実行
filtered_result = filter_jobs_and_levels(players)

from itertools import product
advanced_jobs = ['バトマス', '賢者', 'レンジャ', '魔法戦士', 'パラディ', 'スパスタ', '海賊', 'まもマス']
special_jobs = ['ゴドハン', '大魔導師', '大神官', 'ニンジャ', '魔剣士', '守り人', 'ドラゴン', '天地雷鳴']

def calculate_adjusted_level(job, level):
    # 特級職（インデックス8以降）の場合、レベルを+45する
    return level + 45 if job in special_jobs else level

def find_valid_combinations(filtered_result):
    # 各プレーヤーの職業とレベルの組み合わせを作成
    player_options = []
    for player_id in filtered_result:
        jobs = filtered_result[player_id]['jobs']
        levels = filtered_result[player_id]['levels']
        options = list(zip(jobs, levels))
        player_options.append(options)
    
    valid_combinations = []
    
    # 全ての組み合わせを生成
    for combination in product(*player_options):
        jobs = [job for job, _ in combination]
        levels = [level for _, level in combination]
        
        # 条件1: 平均レベルチェック
        adjusted_levels = [calculate_adjusted_level(job, level) 
                         for job, level in combination]
        avg_level = sum(adjusted_levels) / len(adjusted_levels)
        
        # 条件2: ニンジャチェック
        has_ninja = 'ニンジャ' in jobs
        
        # 条件3: バトマス、魔剣士、ドラゴンのいずれかチェック
        has_required_job = any(job in ['バトマス', '魔剣士', 'ドラゴン'] for job in jobs)
        
        # 条件4: 上級職チェック
        has_advanced_job = any(job in advanced_jobs for job in jobs)
        
        # 全ての条件を満たす場合
        if 95 <= avg_level < 100 and has_ninja and has_required_job and has_advanced_job:
            valid_combinations.append({
                'combination': combination,
                'average_level': avg_level
            })
    
    return valid_combinations

# 有効な組み合わせを見つける
valid_combinations = find_valid_combinations(filtered_result)

# 結果を表示（平均レベルでソート）
sorted_combinations = sorted(valid_combinations, key=lambda x: x['average_level'])

for i, result in enumerate(sorted_combinations, 1):
    print(f"\n組み合わせ {i}:")
    print(f"平均レベル: {result['average_level']:.2f}")
    for player_id, (job, level) in zip(filtered_result.keys(), result['combination']):
        adjusted_level = calculate_adjusted_level(job, level)
        player_name = filtered_result[player_id]['name']
        print(f"{player_name}: {job} (Lv.{level}, 調整後Lv.{adjusted_level})")