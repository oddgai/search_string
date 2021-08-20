import streamlit as st
import numpy as np
import random
from deap import algorithms, base, creator, tools

def random_character_code():
	'''ランダムな文字列のUnicode値を返す'''
	# 半角記号、英数字、ひらがな、カタカナ、だいたいの漢字
	char_code = list(range(32, 126)) + list(range(12289, 12541)) + kanji_unicode_list
	return random.choice(char_code)

def obj_func(x, target_code):
	'''目的関数: 文字列の一致度'''
	match = [i == j for i, j in zip(x, target_code)]
	obj = sum(match) / len(match)
	return obj,  # returnの後に,をつける

### 常用漢字、人名用漢字のリストを読み込む
kanji_list = np.loadtxt("src/kanji_list.txt", dtype=object, delimiter=" ")
kanji_unicode_list = [ord(s) for s in kanji_list]

### 表示用
st.title("遺伝的アルゴリズムで答えを探します")
target_string = st.text_input("答えを教えてね（2文字以上）")
start_btn = st.button("探索スタート")
st.write("---")

current_fitness = st.empty()
search_progress_bar = st.progress(0)
current_string = st.empty()

result = 0  # 正解したかどうか

# パラメータ
NGEN = 100  # 世代数
POP = 10000  # 1世代内の個体数

if start_btn and len(target_string) >= 2:
	with st.spinner("探索中・・・"):

		# 答えを文字コードに変換
		target_code = [ord(s) for s in target_string]
		size = len(target_code)

		### GAの実装
		creator.create("TransformCharacter", base.Fitness, weights=(1.0,))  # 最大化問題として定義
		creator.create("Individual", list, fitness=creator.TransformCharacter)  # 個体Individualをlistとして定義

		# 各種設定
		toolbox = base.Toolbox()
		toolbox.register("g", random_character_code)  # 遺伝子g: 0-1整数
		toolbox.register("x", tools.initRepeat, creator.Individual, toolbox.g, n=size)  # 各個体x: gをsize回くり返したlist
		toolbox.register("population", tools.initRepeat, list, toolbox.x)  # 集団
		toolbox.register("select", tools.selTournament, tournsize=5)  # 選択方式: トーナメント方式
		toolbox.register("mate", tools.cxTwoPoint)  # 交叉関数: 二点交叉
		toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.6)  # 突然変異関数（indpb: 遺伝子の突然変異率）
		toolbox.register("evaluate", obj_func, target_code)  # 目的関数

		# GA実行
		population = toolbox.population(n=POP)
		for gen in range(NGEN):
			offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.6)
			fits = toolbox.map(toolbox.evaluate, offspring)
			for fit, ind in zip(fits, offspring):
				ind.fitness.values = fit
			best_ind = tools.selBest(population, 1)[0]
			best_ind_char = "".join([chr(i) for i in best_ind])
			# print(f"{gen}: {best_ind_char}, {best_ind.fitness.values}")

			# 暫定解を表示
			if best_ind.fitness.values:
				current_string.write(f"### {best_ind_char}")
				current_fitness.caption(f"正解率: {np.int8(best_ind.fitness.values[0] * 100)} %")
				search_progress_bar.progress(best_ind.fitness.values[0])

			# 答えに一致したら表示
			if best_ind.fitness.values == (1,):
				result = 1
				break
			else:
				population = toolbox.select(offspring, k=len(population))

	if result == 1:
		st.success("正解しました！ほめてください！")
	else:
		st.error("だめでした・・・次回にご期待ください！")
