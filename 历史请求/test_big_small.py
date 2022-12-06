import random

# 初始化积分
score = 100

while True:
    # 生成随机数
    number = random.randint(1, 10)

    # 获取用户的输入
    guess = input('请输入您的猜测 (大或小): ')
    bet = int(input('请输入您的下注金额: '))

    # 判断用户的猜测是否正确
    if (guess == '大' and number > 5) or (guess == '小' and number < 5):
        # 猜对了，奖励双倍积分
        score += bet * 2
        print('恭喜您猜对了，当前积分为', score)
    else:
        # 猜错了，失去下注金额
        score -= bet
        print('抱歉您猜错了，当前积分为', score)

    # 判断游戏是否结束
    if score <= 0:
        print('您的积分不足，游戏结束')
        break

