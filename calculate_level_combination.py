import itertools
from tqdm import tqdm

# 上級職と特級職のインデックス範囲を定義
advanced_job_indices = range(0, 8)  # バトマス〜まもマス
special_job_indices = range(8, 16)  # ゴドハン〜天地雷鳴

def find_combinations_for_fixed_jobs(fixed_players, target_player, lower_avg, upper_avg):
    valid_combinations = set()  # セットを使用して重複を防ぐ
    num_players = len(fixed_players) + 1
    
    # 使用済みの職業を追跡
    used_jobs = {player['jobs'][player['fixed_job_index']] for player in fixed_players}
    
    # 固定レベルのタプルを作成（毎回の計算で再利用）
    fixed_levels = tuple(player['fixed_level'] for player in fixed_players)
    
    # ターゲットプレイヤーの全職業とレベルを処理
    for job_index, (level, job) in enumerate(zip(target_player['levels'], target_player['jobs'])):
        # 既に使用されている職業はスキップ
        if job in used_jobs:
            continue
            
        # レベル制限チェック
        if (job_index in advanced_job_indices and level >= 75) or \
           (job_index in special_job_indices and level >= 70):
            continue
            
        # 上級職チェック
        has_advanced_job = any(
            fixed_player['fixed_job_index'] in advanced_job_indices 
            for fixed_player in fixed_players
        ) or (job_index in advanced_job_indices)
        
        if not has_advanced_job:
            continue
            
        # ニンジャチェック
        has_ninja = any(
            fixed_player['jobs'][fixed_player['fixed_job_index']] == 'ニンジャ'
            for fixed_player in fixed_players
        ) or (job == 'ニンジャ')
        
        if not has_ninja:
            continue
            
        # ゴドハン、ドラゴン、魔剣士チェック
        required_jobs = {'ゴドハン', 'ドラゴン', '魔剣士'}
        has_required_job = any(
            fixed_player['jobs'][fixed_player['fixed_job_index']] in required_jobs
            for fixed_player in fixed_players
        ) or (job in required_jobs)
        
        if not has_required_job:
            continue
            
        # 特級職の補正を適用してレベル平均を計算
        adjusted_levels = list(fixed_levels)  # 固定プレイヤーのレベル
        
        # 固定プレイヤーの補正
        for i, player in enumerate(fixed_players):
            if player['fixed_job_index'] in special_job_indices:
                adjusted_levels[i] += 45
                
        # 可変プレイヤーの補正
        final_level = level + 45 if job_index in special_job_indices else level
        adjusted_levels.append(final_level)
        
        avg = sum(adjusted_levels) / num_players
        
        if lower_avg <= avg <= upper_avg:
            # ユニークな識別子として、レベルの組み合わせと職業を含むタプルを作成
            combination_key = (avg, fixed_levels + (level,), job)
            valid_combinations.add(combination_key)
    
    return sorted(valid_combinations)  # ソートされたリストとして返す

def try_all_fixed_combinations():
    max_combinations = 0
    best_result = None
    
    # プレーヤー2を天地雷鳴（インデックス15）で固定
    player2_fixed = {
        'fixed_level': players['player2']['levels'][15],
        'fixed_job_index': 15,
        'jobs': players['player2']['jobs']
    }
    
    # プレーヤー2以外のプレーヤーから2人を固定して1人を可変にする
    remaining_players = ['player1', 'player3', 'player4']
    for variable_idx in range(len(remaining_players)):
        fixed_indices = [i for i in range(len(remaining_players)) if i != variable_idx]
        
        # 固定する2人の全職業の組み合わせを試す
        for fixed_jobs in itertools.product(range(16), repeat=2):
            fixed_players = [player2_fixed]  # プレーヤー2を最初に追加
            
            # 残りの固定プレイヤーを追加
            for idx, job_idx in zip(fixed_indices, fixed_jobs):
                player = players[remaining_players[idx]]
                fixed_players.append({
                    'fixed_level': player['levels'][job_idx],
                    'fixed_job_index': job_idx,
                    'jobs': player['jobs']
                })
            
            target_player = players[remaining_players[variable_idx]]
            
            results = find_combinations_for_fixed_jobs(
                fixed_players, target_player, lower_avg, upper_avg)
                
            if results and len(results) > max_combinations:
                max_combinations = len(results)
                best_result = {
                    'fixed_players': [
                        {
                            'name': players[remaining_players[idx]]['name'],
                            'job': players[remaining_players[idx]]['jobs'][job_idx],
                            'level': players[remaining_players[idx]]['levels'][job_idx]
                        }
                        for idx, job_idx in zip(fixed_indices, fixed_jobs)
                    ],
                    'variable_player': players[remaining_players[variable_idx]]['name'],
                    'combinations': results
                }
    
    return best_result

def display_best_result(result):
    if not result:
        print("条件を満たす組み合わせが見つかりませんでした。")
        return
        
    print("\n最も多くの組み合わせが存在するパターン:")
    print("\n固定プレイヤー:")
    for player in result['fixed_players']:
        print(f"{player['name']}: {player['job']} (Lv.{player['level']})")
    print(f"\n可変プレイヤー: {result['variable_player']}")
    print(f"\n組み合わせ数: {len(result['combinations'])}件\n")
    
    # 組み合わせの詳細を表示
    sorted_combinations = sorted(result['combinations'], key=lambda x: x[0])
    for avg, levels, job in sorted_combinations:
        print(f"平均レベル: {avg:.1f} ({job})")
        print(f"レベル組み合わせ: {levels}")
        print("-" * 50)

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
        'levels': [75, 72, 89, 74, 72, 72, 80, 72, 65, 58, 58, 1, 68, 72, 56, 37],
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

# 平均値の閾値を設定
lower_avg = 95  # 平均レベル下限
upper_avg = 96  # 平均レベル上限

# 最適な組み合わせを探して表示
best_result = try_all_fixed_combinations()
display_best_result(best_result)