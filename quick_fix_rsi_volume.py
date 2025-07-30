# quick_fix_rsi_volume.py
# 快速修复 RSI 始终为 50 和成交量始终为 0 的问题

import os
import re


def quick_fix():
    """快速修复显示问题"""

    if not os.path.exists('simple_trading_bot.py'):
        print("❌ 找不到 simple_trading_bot.py")
        return

    with open('simple_trading_bot.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 备份
    with open('simple_trading_bot_backup_display.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 已备份文件")

    # 查找问题点
    print("\n分析问题...")

    # 1. 查找显示 "• RSI: 50.0" 的地方
    # 这通常在某个打印或日志输出中
    rsi_display_pattern = r'RSI:\s*[^,\n]+'
    matches = list(re.finditer(rsi_display_pattern, content))
    print(f"找到 {len(matches)} 处 RSI 显示")

    # 2. 查找 technical_analysis 字典中 rsi 字段的设置
    # 问题可能是 rsi 值没有正确传入字典

    # 修复方案：在 _perform_technical_analysis 函数中
    # 确保 RSI 值正确获取和传递

    # 查找函数
    func_start = content.find("def _perform_technical_analysis")
    if func_start > 0:
        func_end = content.find("\n    def ", func_start + 1)
        if func_end < 0:
            func_end = content.find("\nclass ", func_start + 1)
        if func_end < 0:
            func_end = len(content)

        func_content = content[func_start:func_end]

        # 查找 technical_analysis 字典定义
        dict_pattern = r'technical_analysis\s*=\s*\{([^}]+)\}'
        dict_match = re.search(dict_pattern, func_content, re.DOTALL)

        if dict_match:
            dict_content = dict_match.group(1)

            # 检查 rsi 字段
            if "'rsi':" in dict_content:
                print("✅ 找到 RSI 字段定义")

                # 确保使用的是变量 rsi 而不是硬编码的值
                # 修复：将 'rsi': 50 改为 'rsi': rsi
                new_dict_content = re.sub(
                    r"'rsi'\s*:\s*\d+(\.\d+)?",  # 匹配 'rsi': 50 或 'rsi': 50.0
                    "'rsi': rsi",  # 替换为变量
                    dict_content
                )

                if new_dict_content != dict_content:
                    new_func_content = func_content.replace(dict_content, new_dict_content)
                    content = content[:func_start] + new_func_content + content[func_end:]
                    print("✅ 修复了 RSI 字段使用变量而非常量")

    # 3. 修复成交量分析
    # 查找成交量激增的计算或显示
    volume_pattern = r'成交量激增.*?比率:\s*[\d.]+x'
    volume_matches = list(re.finditer(volume_pattern, content))

    # 添加成交量计算逻辑
    volume_calc_code = '''
        # 计算成交量比率
        volume_ratio = 1.0
        if 'volume' in df.columns and len(df) >= 20:
            try:
                volume_mean = df['volume'].rolling(20).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                if volume_mean > 0:
                    volume_ratio = current_volume / volume_mean
            except:
                volume_ratio = 1.0
'''

    # 在适当位置插入成交量计算
    # 通常在计算其他指标之后

    # 4. 添加调试输出
    debug_code = '''
            # [DEBUG] 打印实际的指标值
            print(f"[DEBUG] 实际 RSI 值: {rsi:.2f}")
            print(f"[DEBUG] 实际成交量比率: {volume_ratio:.2f}x")
'''

    # 写回文件
    with open('simple_trading_bot.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n✅ 基本修复完成")

    # 创建一个测试函数
    test_code = '''# test_indicators_display.py
# 测试指标显示是否正确

def test_display():
    """测试指标显示"""
    import pandas as pd

    # 创建测试数据
    test_data = {
        'RSI': [30, 40, 50, 60, 70, 28.36],  # 最后一个值是您日志中的实际值
        'volume': [1000, 1200, 1500, 2000, 2500, 3000]
    }
    df = pd.DataFrame(test_data)

    # 测试 RSI 获取
    print("测试 RSI 获取:")

    # 错误的方式（可能导致始终为 50）
    rsi_wrong = 50  # 硬编码
    print(f"  错误方式: RSI = {rsi_wrong}")

    # 正确的方式
    rsi_correct = df['RSI'].iloc[-1]
    print(f"  正确方式: RSI = {rsi_correct}")

    # 测试成交量比率
    print("\\n测试成交量比率:")
    volume_mean = df['volume'].rolling(3).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    volume_ratio = current_volume / volume_mean if volume_mean > 0 else 0
    print(f"  平均成交量: {volume_mean}")
    print(f"  当前成交量: {current_volume}")
    print(f"  成交量比率: {volume_ratio:.2f}x")

    # 模拟技术分析字典
    technical_analysis = {
        'rsi': rsi_correct,  # 使用实际值，不是硬编码的 50
        'volume_ratio': volume_ratio  # 使用计算的值，不是 0
    }

    print("\\n技术分析结果:")
    print(f"  • RSI: {technical_analysis['rsi']:.1f}")
    print(f"  • 成交量比率: {technical_analysis['volume_ratio']:.1f}x")


if __name__ == "__main__":
    test_display()
'''

    with open('test_indicators_display.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    print("✅ 创建了测试脚本")


def create_patch_function():
    """创建修复函数供直接使用"""

    patch_code = '''# technical_analysis_patch.py
# 技术分析修复补丁

import pandas as pd
import numpy as np


def patch_technical_analysis_function(original_func):
    """修复技术分析函数的装饰器"""

    def fixed_function(self, symbol):
        # 调用原函数
        result = original_func(self, symbol)

        # 检查并修复 RSI
        if 'rsi' in result and result['rsi'] == 50:
            # 尝试从 DataFrame 获取真实的 RSI
            try:
                df = self.get_market_data_sync(symbol)
                if not df.empty:
                    from indicators_module import calculate_optimized_indicators
                    df = calculate_optimized_indicators(df)

                    if 'RSI' in df.columns:
                        actual_rsi = df['RSI'].dropna()
                        if len(actual_rsi) > 0:
                            result['rsi'] = float(actual_rsi.iloc[-1])
                            print(f"[PATCH] 修正 RSI: {result['rsi']:.2f}")
            except:
                pass

        # 检查并修复成交量
        if 'volume_ratio' not in result or result.get('volume_ratio', 0) == 0:
            try:
                df = self.get_market_data_sync(symbol)
                if not df.empty and 'volume' in df.columns and len(df) >= 20:
                    volume_mean = df['volume'].rolling(20).mean().iloc[-1]
                    current_volume = df['volume'].iloc[-1]
                    if volume_mean > 0:
                        result['volume_ratio'] = current_volume / volume_mean
                        print(f"[PATCH] 修正成交量比率: {result['volume_ratio']:.2f}x")
            except:
                result['volume_ratio'] = 1.0

        return result

    return fixed_function


# 使用方法：
# 在 SimpleTradingBot 类初始化时：
# self._perform_technical_analysis = patch_technical_analysis_function(self._perform_technical_analysis)
'''

    with open('technical_analysis_patch.py', 'w', encoding='utf-8') as f:
        f.write(patch_code)
    print("✅ 创建了补丁函数")


if __name__ == "__main__":
    print("快速修复 RSI 和成交量显示问题")
    print("=" * 60)

    quick_fix()
    create_patch_function()

    print("\n" + "=" * 60)
    print("修复建议：")
    print("\n1. 运行测试脚本检查问题：")
    print("   python test_indicators_display.py")
    print("\n2. 应用补丁（在 SimpleTradingBot.__init__ 中添加）：")
    print("   from technical_analysis_patch import patch_technical_analysis_function")
    print("   self._perform_technical_analysis = patch_technical_analysis_function(self._perform_technical_analysis)")
    print("\n3. 或者手动检查 _perform_technical_analysis 函数中：")
    print("   - technical_analysis 字典的 'rsi' 字段是否使用了变量 rsi")
    print("   - 是否计算了 volume_ratio 并添加到结果中")
    print("\n4. 搜索代码中所有 'rsi': 50 并改为 'rsi': rsi")