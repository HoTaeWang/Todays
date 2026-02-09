"""
Day 1 실습: NLTK Book 탐험
실행 방법: python day1_practice.py
"""

from nltk.book import *

print("=" * 70)
print("🎮 Day 1 실습: NLTK Book 탐험 시작!")
print("=" * 70)

# ============================================================================
# Part 1: 텍스트 목록 확인
# ============================================================================
print("\n📚 Part 1: 사용 가능한 텍스트 확인")
print("-" * 70)

print("text1:", text1)
print("text2:", text2)
print("text3:", text3)
print("text4:", text4)
print("text5:", text5)

# ============================================================================
# Part 2: Concordance - 단어의 문맥 찾기
# ============================================================================
print("\n🔍 Part 2: Concordance - 'whale' 단어의 문맥")
print("-" * 70)

text1.concordance("whale", lines=5)

# ============================================================================
# Part 3: Similar - 비슷한 맥락의 단어
# ============================================================================
print("\n🎯 Part 3: Similar - 'monstrous'와 비슷한 단어들")
print("-" * 70)

text1.similar("monstrous")

# ============================================================================
# Part 4: Common Contexts - 공통 문맥
# ============================================================================
print("\n🔗 Part 4: Common Contexts - 'ship'과 'boat'의 공통 문맥")
print("-" * 70)

text1.common_contexts(["ship", "boat"])

# ============================================================================
# Part 5: 기본 통계
# ============================================================================
print("\n📊 Part 5: Text1 (Moby Dick) 기본 통계")
print("-" * 70)

total_words = len(text1)
unique_words = len(set(text1))
lexical_diversity = unique_words / total_words

print(f"총 단어 수: {total_words:,}")
print(f"고유 단어 수: {unique_words:,}")
print(f"어휘 다양성: {lexical_diversity:.4f}")

whale_count = text1.count("whale")
whale_percentage = 100 * whale_count / total_words
print(f"\n'whale' 빈도수: {whale_count}회")
print(f"'whale' 비율: {whale_percentage:.4f}%")

# ============================================================================
# Part 6: 미션 1-1 - 단어 탐정
# ============================================================================
print("\n" + "=" * 70)
print("🎮 미션 1-1: 단어 탐정 - 'captain' 조사하기")
print("=" * 70)

print("\n1. 'captain'의 문맥:")
text1.concordance("captain", lines=5)

print("\n2. 'captain'과 비슷한 단어들:")
text1.similar("captain")

print("\n3. 'captain' 빈도수:")
captain_count = text1.count("captain")
print(f"   'captain' 등장 횟수: {captain_count}회")

# ============================================================================
# Part 7: 미션 1-2 - 텍스트 비교 분석
# ============================================================================
print("\n" + "=" * 70)
print("🎮 미션 1-2: 텍스트 비교 - text1 vs text4")
print("=" * 70)

# text1: Moby Dick (소설)
text1_total = len(text1)
text1_unique = len(set(text1))
text1_diversity = text1_unique / text1_total

# text4: Inaugural Addresses (연설문)
text4_total = len(text4)
text4_unique = len(set(text4))
text4_diversity = text4_unique / text4_total

print("\n[Text1 - Moby Dick (소설)]")
print(f"  총 단어 수: {text1_total:,}")
print(f"  고유 단어 수: {text1_unique:,}")
print(f"  어휘 다양성: {text1_diversity:.4f}")

print("\n[Text4 - Inaugural Addresses (연설문)]")
print(f"  총 단어 수: {text4_total:,}")
print(f"  고유 단어 수: {text4_unique:,}")
print(f"  어휘 다양성: {text4_diversity:.4f}")

print("\n[분석 결과]")
if text1_diversity > text4_diversity:
    print(f"  → Text1(소설)이 Text4(연설문)보다 {text1_diversity - text4_diversity:.4f} 더 다양한 어휘 사용")
else:
    print(f"  → Text4(연설문)이 Text1(소설)보다 {text4_diversity - text1_diversity:.4f} 더 다양한 어휘 사용")

# ============================================================================
# Part 8: 미션 1-3 - 패턴 발견
# ============================================================================
print("\n" + "=" * 70)
print("🎮 미션 1-3: 패턴 발견 - text2 (Sense and Sensibility)")
print("=" * 70)

print("\n1. 'love'와 'hate'의 공통 문맥:")
text2.common_contexts(["love", "hate"])

print("\n2. 'Mr'와 'Mrs'의 공통 문맥:")
text2.common_contexts(["Mr", "Mrs"])

love_count = text2.count("love")
hate_count = text2.count("hate")
mr_count = text2.count("Mr")
mrs_count = text2.count("Mrs")

print("\n[빈도 분석]")
print(f"  'love': {love_count}회")
print(f"  'hate': {hate_count}회")
print(f"  'Mr': {mr_count}회")
print(f"  'Mrs': {mrs_count}회")

# ============================================================================
# Part 9: 추가 도전 과제 - 9개 텍스트 비교
# ============================================================================
print("\n" + "=" * 70)
print("🏆 추가 도전 과제: 9개 텍스트 어휘 다양성 비교")
print("=" * 70)

texts_dict = {
    "Text1: Moby Dick": text1,
    "Text2: Sense & Sensibility": text2,
    "Text3: Genesis": text3,
    "Text4: Inaugural": text4,
    "Text5: Chat": text5,
    "Text6: Monty Python": text6,
    "Text7: Wall Street Journal": text7,
    "Text8: Personals": text8,
    "Text9: Chesterton": text9
}

diversity_results = []

print("\n어휘 다양성 순위:")
print("-" * 70)

for name, text in texts_dict.items():
    diversity = len(set(text)) / len(text)
    diversity_results.append((name, diversity, len(text), len(set(text))))

# 다양성 순으로 정렬
diversity_results.sort(key=lambda x: x[1], reverse=True)

for rank, (name, diversity, total, unique) in enumerate(diversity_results, 1):
    print(f"{rank}. {name}")
    print(f"   어휘 다양성: {diversity:.4f} (전체: {total:,}, 고유: {unique:,})")

print("\n[분석]")
print(f"가장 다양: {diversity_results[0][0]} ({diversity_results[0][1]:.4f})")
print(f"가장 단순: {diversity_results[-1][0]} ({diversity_results[-1][1]:.4f})")

# ============================================================================
# 완료!
# ============================================================================
print("\n" + "=" * 70)
print("✅ Day 1 실습 완료!")
print("=" * 70)
print("\n획득 경험치:")
print("  - 일일 과제: +10 XP")
print("  - 미션 1-1: +50 XP")
print("  - 미션 1-2: +50 XP")
print("  - 미션 1-3: +50 XP")
print("  - 추가 도전: +25 XP")
print("  ────────────────")
print("  총 획득: +185 XP")
print("\n진행도: 185 / 200 XP (Level 2까지 15 XP 남음!)")
print("\n다음: Day 2 - Tokenization (토큰화) 기초")
print("=" * 70)
