"""
增强版市场微观结构博弈论分析系统
真正捕捉庄家意图，结合现货和合约数据
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from logger_utils import Colors, print_colored


class EnhancedGameTheoryAnalyzer:
    """
    增强版博弈论分析器
    核心功能：
    1. 订单簿深度分析（识别冰山单、支撑阻力）
    2. 现货大单追踪（影响合约价格）
    3. 资金流向分析
    4. 庄家行为模式识别
    5. 技术指标融合
    """

    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger('EnhancedGameTheoryAnalyzer')

        # 分析参数
        self.params = {
            'iceberg_threshold': 0.3,      # 冰山单检测阈值
            'whale_order_threshold': 50000, # 大单阈值（USDT）
            'order_book_depth': 20,        # 订单簿深度
            'spot_futures_correlation': 0.8 # 现货期货相关性阈值
        }

        print_colored("✅ 增强版博弈论分析器初始化完成", Colors.GREEN)

    def analyze_market_intent(self, symbol: str) -> Dict[str, Any]:
        """
        综合分析市场意图和庄家行为
        """
        print_colored(f"\n🔍 深度分析 {symbol} 市场结构...", Colors.CYAN)

        analysis_result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'whale_intent': 'NEUTRAL',
            'confidence': 0.0,
            'signals': [],
            'risk_factors': [],
            'technical_confluence': {},
            'recommendation': 'HOLD'
        }

        try:
            # 1. 获取并分析订单簿
            print_colored("  📊 分析订单簿结构...", Colors.INFO)
            order_book_analysis = self._analyze_order_book(symbol)
            if order_book_analysis:
                analysis_result['order_book'] = order_book_analysis
                self._log_order_book_insights(order_book_analysis)

            # 2. 分析现货市场大单
            print_colored("  🐋 追踪现货大单流向...", Colors.INFO)
            spot_flow = self._analyze_spot_whale_trades(symbol.replace('USDT', '') + 'USDT')
            if spot_flow:
                analysis_result['spot_flow'] = spot_flow
                self._log_spot_flow_insights(spot_flow)

            # 3. 获取并分析资金费率
            print_colored("  💰 分析资金费率和持仓...", Colors.INFO)
            funding_analysis = self._analyze_funding_and_oi(symbol)
            if funding_analysis:
                analysis_result['funding'] = funding_analysis
                self._log_funding_insights(funding_analysis)

            # 4. 技术指标验证
            print_colored("  📈 计算技术指标共振...", Colors.INFO)
            technical_signals = self._get_technical_confluence(symbol)
            if technical_signals:
                analysis_result['technical_confluence'] = technical_signals
                self._log_technical_insights(technical_signals)

            # 5. 综合判断庄家意图
            print_colored("  🧠 综合判断市场意图...", Colors.INFO)
            whale_intent = self._determine_whale_intent(
                order_book_analysis,
                spot_flow,
                funding_analysis,
                technical_signals
            )

            analysis_result.update(whale_intent)

            # 6. 生成交易建议
            if whale_intent['confidence'] > 0.5:
                if whale_intent['whale_intent'] == 'ACCUMULATION':
                    analysis_result['recommendation'] = 'BUY'
                    analysis_result['signals'].append("🟢 庄家吸筹信号")
                elif whale_intent['whale_intent'] == 'DISTRIBUTION':
                    analysis_result['recommendation'] = 'SELL'
                    analysis_result['signals'].append("🔴 庄家派发信号")
                elif whale_intent['whale_intent'] == 'MANIPULATION_UP':
                    analysis_result['recommendation'] = 'BUY_CAUTIOUS'
                    analysis_result['signals'].append("⚠️ 疑似拉盘操纵")
                elif whale_intent['whale_intent'] == 'MANIPULATION_DOWN':
                    analysis_result['recommendation'] = 'SELL_CAUTIOUS'
                    analysis_result['signals'].append("⚠️ 疑似砸盘操纵")

            # 打印最终判断
            self._log_final_verdict(analysis_result)

        except Exception as e:
            self.logger.error(f"分析{symbol}失败: {e}")
            print_colored(f"  ❌ 分析出错: {str(e)}", Colors.ERROR)
            analysis_result['error'] = str(e)

        return analysis_result

    def _analyze_order_book(self, symbol: str) -> Dict[str, Any]:
        """
        深度分析订单簿，识别关键特征
        """
        try:
            # 获取订单簿
            order_book = self.client.futures_order_book(symbol=symbol, limit=1000)

            bids = [(float(price), float(qty)) for price, qty in order_book['bids']]
            asks = [(float(price), float(qty)) for price, qty in order_book['asks']]

            if not bids or not asks:
                return None

            current_price = (bids[0][0] + asks[0][0]) / 2

            # 分析买卖压力
            bid_volume = sum(qty for _, qty in bids[:20])
            ask_volume = sum(qty for _, qty in asks[:20])
            pressure_ratio = bid_volume / ask_volume if ask_volume > 0 else 0

            # 检测冰山单
            iceberg_orders = self._detect_iceberg_orders(bids, asks)

            # 识别支撑阻力墙
            support_walls = self._find_order_walls(bids, 'support')
            resistance_walls = self._find_order_walls(asks, 'resistance')

            # 计算订单簿失衡度
            imbalance = self._calculate_order_book_imbalance(bids, asks)

            # 分析订单分布
            bid_distribution = self._analyze_order_distribution(bids)
            ask_distribution = self._analyze_order_distribution(asks)

            analysis = {
                'current_price': current_price,
                'pressure_ratio': pressure_ratio,
                'bid_volume_20': bid_volume,
                'ask_volume_20': ask_volume,
                'imbalance': imbalance,
                'iceberg_orders': iceberg_orders,
                'support_walls': support_walls,
                'resistance_walls': resistance_walls,
                'bid_distribution': bid_distribution,
                'ask_distribution': ask_distribution
            }

            return analysis

        except Exception as e:
            self.logger.error(f"订单簿分析失败: {e}")
            return None

    def _detect_iceberg_orders(self, bids: List[Tuple[float, float]],
                              asks: List[Tuple[float, float]]) -> Dict[str, List[Dict]]:
        """
        检测冰山单（隐藏的大额订单）
        """
        iceberg_orders = {'buy': [], 'sell': []}

        # 检测买单中的冰山单
        for i in range(len(bids) - 1):
            price, qty = bids[i]

            # 检查相邻价位是否有相似数量的订单（冰山单特征）
            similar_qty_count = 0
            for j in range(max(0, i-3), min(len(bids), i+4)):
                if i != j and abs(bids[j][1] - qty) / qty < 0.1:  # 数量相差10%以内
                    similar_qty_count += 1

            if similar_qty_count >= 2:  # 至少有2个相似订单
                iceberg_orders['buy'].append({
                    'price': price,
                    'visible_qty': qty,
                    'estimated_total': qty * (similar_qty_count + 1),
                    'confidence': min(similar_qty_count * 0.25, 0.9)
                })

        # 检测卖单中的冰山单（类似逻辑）
        for i in range(len(asks) - 1):
            price, qty = asks[i]
            similar_qty_count = 0
            for j in range(max(0, i-3), min(len(asks), i+4)):
                if i != j and abs(asks[j][1] - qty) / qty < 0.1:
                    similar_qty_count += 1

            if similar_qty_count >= 2:
                iceberg_orders['sell'].append({
                    'price': price,
                    'visible_qty': qty,
                    'estimated_total': qty * (similar_qty_count + 1),
                    'confidence': min(similar_qty_count * 0.25, 0.9)
                })

        return iceberg_orders

    def _find_order_walls(self, orders: List[Tuple[float, float]],
                         wall_type: str) -> List[Dict[str, Any]]:
        """
        识别订单墙（大额挂单）
        """
        if not orders:
            return []

        # 计算平均订单量
        avg_qty = sum(qty for _, qty in orders[:50]) / min(50, len(orders))

        walls = []
        for price, qty in orders[:20]:  # 只看前20档
            if qty > avg_qty * 5:  # 超过平均值5倍视为墙
                walls.append({
                    'price': price,
                    'quantity': qty,
                    'strength': qty / avg_qty,
                    'type': wall_type
                })

        # 按强度排序
        walls.sort(key=lambda x: x['strength'], reverse=True)
        return walls[:3]  # 返回最强的3个墙

    def _analyze_spot_whale_trades(self, spot_symbol: str) -> Dict[str, Any]:
        """
        分析现货市场的大单交易
        """
        try:
            # 获取最近的成交
            trades = self.client.get_recent_trades(symbol=spot_symbol, limit=1000)

            # 转换为DataFrame便于分析
            df = pd.DataFrame(trades)
            df['price'] = df['price'].astype(float)
            df['qty'] = df['qty'].astype(float)
            df['quoteQty'] = df['quoteQty'].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='ms')

            # 识别大单
            whale_threshold = self.params['whale_order_threshold']
            df['is_whale'] = df['quoteQty'] > whale_threshold

            # 统计大单买卖
            whale_trades = df[df['is_whale']]

            if len(whale_trades) == 0:
                return {
                    'whale_buy_volume': 0,
                    'whale_sell_volume': 0,
                    'whale_net_flow': 0,
                    'whale_trades_count': 0
                }

            # 判断买卖方向（这里简化处理，实际需要更复杂的逻辑）
            # 使用 .loc 来避免 SettingWithCopyWarning
            whale_trades.loc[:, 'is_buy'] = whale_trades['isBuyerMaker'] == False

            whale_buy_volume = whale_trades[whale_trades['is_buy']]['quoteQty'].sum()
            whale_sell_volume = whale_trades[~whale_trades['is_buy']]['quoteQty'].sum()

            # 计算最近的大单趋势
            recent_whales = whale_trades.tail(10)
            recent_buy_count = len(recent_whales[recent_whales['is_buy']])
            recent_sell_count = len(recent_whales) - recent_buy_count

            analysis = {
                'whale_buy_volume': whale_buy_volume,
                'whale_sell_volume': whale_sell_volume,
                'whale_net_flow': whale_buy_volume - whale_sell_volume,
                'whale_trades_count': len(whale_trades),
                'total_trades_count': len(df),
                'whale_ratio': len(whale_trades) / len(df),
                'recent_whale_trend': 'BUY' if recent_buy_count > recent_sell_count else 'SELL',
                'largest_trades': whale_trades.nlargest(5, 'quoteQty')[['price', 'qty', 'quoteQty', 'is_buy']].to_dict('records')
            }

            return analysis

        except Exception as e:
            self.logger.error(f"现货大单分析失败: {e}")
            return None

    def _analyze_funding_and_oi(self, symbol: str) -> Dict[str, Any]:
        """
        分析资金费率和持仓量变化
        """
        try:
            # 获取资金费率
            funding_rate = self.client.futures_funding_rate(symbol=symbol, limit=1)
            current_funding = float(funding_rate[0]['fundingRate']) if funding_rate else 0

            # 获取持仓量
            oi_stats = self.client.futures_open_interest(symbol=symbol)
            current_oi = float(oi_stats['openInterest'])

            # 获取历史数据对比
            hist_oi = self.client.futures_open_interest_hist(
                symbol=symbol,
                period='5m',
                limit=12  # 1小时数据
            )

            if hist_oi:
                oi_1h_ago = float(hist_oi[0]['sumOpenInterest'])
                oi_change = (current_oi - oi_1h_ago) / oi_1h_ago if oi_1h_ago > 0 else 0
            else:
                oi_change = 0

            # 获取多空比（如果可用）
            try:
                long_short_ratio = self.client.futures_top_longshort_position_ratio(
                    symbol=symbol,
                    period='5m',
                    limit=1
                )
                if long_short_ratio:
                    ls_ratio = float(long_short_ratio[0]['longShortRatio'])
                else:
                    ls_ratio = 1.0
            except:
                ls_ratio = 1.0

            analysis = {
                'funding_rate': current_funding,
                'funding_direction': 'LONG' if current_funding > 0 else 'SHORT',
                'open_interest': current_oi,
                'oi_change_1h': oi_change,
                'long_short_ratio': ls_ratio,
                'market_sentiment': self._interpret_funding_oi(current_funding, oi_change, ls_ratio)
            }

            return analysis

        except Exception as e:
            self.logger.error(f"资金费率分析失败: {e}")
            return None

    def _get_technical_confluence(self, symbol: str) -> Dict[str, Any]:
        """
        获取技术指标共振信号
        """
        try:
            # 获取K线数据
            klines = self.client.futures_klines(symbol=symbol, interval='15m', limit=100)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                               'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                               'taker_buy_quote', 'ignore'])

            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # 计算基础技术指标
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'], 14)

            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['histogram'] = df['macd'] - df['signal']

            # 布林带
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            # 成交量分析
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # 获取最新值
            latest = df.iloc[-1]

            # 判断技术信号
            signals = {
                'rsi': latest['rsi'],
                'rsi_signal': 'OVERBOUGHT' if latest['rsi'] > 70 else 'OVERSOLD' if latest['rsi'] < 30 else 'NEUTRAL',
                'macd_cross': 'BULLISH' if latest['histogram'] > 0 and df.iloc[-2]['histogram'] <= 0 else
                              'BEARISH' if latest['histogram'] < 0 and df.iloc[-2]['histogram'] >= 0 else 'NONE',
                'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']),
                'volume_surge': latest['volume_ratio'] > 2,
                'trend_strength': abs(latest['macd']) / latest['close'] * 100
            }

            # 计算综合技术评分
            tech_score = 0
            if signals['rsi_signal'] == 'OVERSOLD':
                tech_score += 1
            elif signals['rsi_signal'] == 'OVERBOUGHT':
                tech_score -= 1

            if signals['macd_cross'] == 'BULLISH':
                tech_score += 1
            elif signals['macd_cross'] == 'BEARISH':
                tech_score -= 1

            if signals['bb_position'] < 0.2:
                tech_score += 0.5
            elif signals['bb_position'] > 0.8:
                tech_score -= 0.5

            if signals['volume_surge']:
                tech_score = tech_score * 1.5  # 成交量确认

            signals['technical_score'] = tech_score
            signals['current_price'] = latest['close']

            return signals

        except Exception as e:
            self.logger.error(f"技术指标分析失败: {e}")
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _determine_whale_intent(self, order_book: Dict, spot_flow: Dict,
                               funding: Dict, technical: Dict) -> Dict[str, Any]:
        """
        综合判断庄家意图
        """
        intent_scores = {
            'ACCUMULATION': 0,      # 吸筹
            'DISTRIBUTION': 0,      # 派发
            'MANIPULATION_UP': 0,   # 拉盘操纵
            'MANIPULATION_DOWN': 0, # 砸盘操纵
            'NEUTRAL': 0
        }

        confidence = 0.0
        signals = []

        # 1. 订单簿分析
        if order_book:
            # 买压强于卖压
            if order_book['pressure_ratio'] > 1.5:
                intent_scores['ACCUMULATION'] += 1
                signals.append(f"买压强劲 ({order_book['pressure_ratio']:.2f})")
            elif order_book['pressure_ratio'] < 0.7:
                intent_scores['DISTRIBUTION'] += 1
                signals.append(f"卖压强劲 ({order_book['pressure_ratio']:.2f})")

            # 冰山单分析
            if order_book['iceberg_orders']['buy']:
                intent_scores['ACCUMULATION'] += 1.5
                signals.append(f"发现买方冰山单 ({len(order_book['iceberg_orders']['buy'])}个)")
            if order_book['iceberg_orders']['sell']:
                intent_scores['DISTRIBUTION'] += 1.5
                signals.append(f"发现卖方冰山单 ({len(order_book['iceberg_orders']['sell'])}个)")

            # 订单墙分析
            if order_book['support_walls']:
                strongest_support = order_book['support_walls'][0]
                if strongest_support['strength'] > 10:
                    intent_scores['MANIPULATION_UP'] += 1
                    signals.append(f"强支撑墙 @ ${strongest_support['price']:.4f}")

            if order_book['resistance_walls']:
                strongest_resistance = order_book['resistance_walls'][0]
                if strongest_resistance['strength'] > 10:
                    intent_scores['MANIPULATION_DOWN'] += 1
                    signals.append(f"强阻力墙 @ ${strongest_resistance['price']:.4f}")

        # 2. 现货大单分析
        if spot_flow and spot_flow['whale_trades_count'] > 0:
            net_flow = spot_flow['whale_net_flow']
            if net_flow > 100000:  # 净流入超过10万USDT
                intent_scores['ACCUMULATION'] += 2
                signals.append(f"现货大单净流入 ${net_flow:,.0f}")
            elif net_flow < -100000:
                intent_scores['DISTRIBUTION'] += 2
                signals.append(f"现货大单净流出 ${abs(net_flow):,.0f}")

            # 最近趋势
            if spot_flow['recent_whale_trend'] == 'BUY':
                intent_scores['ACCUMULATION'] += 0.5
            else:
                intent_scores['DISTRIBUTION'] += 0.5

        # 3. 资金费率和持仓分析
        if funding:
            # 资金费率分析
            if abs(funding['funding_rate']) > 0.001:  # 0.1%
                if funding['funding_rate'] > 0:
                    intent_scores['MANIPULATION_UP'] += 0.5
                    signals.append(f"高正资金费率 ({funding['funding_rate']:.4%})")
                else:
                    intent_scores['MANIPULATION_DOWN'] += 0.5
                    signals.append(f"高负资金费率 ({funding['funding_rate']:.4%})")

            # 持仓量变化
            oi_change = funding['oi_change_1h']
            if abs(oi_change) > 0.05:  # 5%变化
                if oi_change > 0:
                    intent_scores['ACCUMULATION'] += 1
                    signals.append(f"持仓量增加 {oi_change:.1%}")
                else:
                    intent_scores['DISTRIBUTION'] += 1
                    signals.append(f"持仓量减少 {abs(oi_change):.1%}")

        # 4. 技术指标验证
        if technical:
            tech_score = technical['technical_score']
            if tech_score > 1:
                intent_scores['ACCUMULATION'] += tech_score * 0.5
                signals.append("技术指标看多")
            elif tech_score < -1:
                intent_scores['DISTRIBUTION'] += abs(tech_score) * 0.5
                signals.append("技术指标看空")

            # RSI极值
            if technical['rsi_signal'] == 'OVERSOLD':
                intent_scores['MANIPULATION_DOWN'] += 0.5
                signals.append(f"RSI超卖 ({technical['rsi']:.1f})")
            elif technical['rsi_signal'] == 'OVERBOUGHT':
                intent_scores['MANIPULATION_UP'] += 0.5
                signals.append(f"RSI超买 ({technical['rsi']:.1f})")

        # 确定最终意图
        max_intent = max(intent_scores.items(), key=lambda x: x[1])
        whale_intent = max_intent[0]

        # 计算置信度
        total_score = sum(intent_scores.values())
        if total_score > 0:
            confidence = max_intent[1] / total_score
            # 考虑次高分数，如果太接近则降低置信度
            sorted_scores = sorted(intent_scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[1] > 0:
                score_diff = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
                confidence *= (0.5 + score_diff * 0.5)

        return {
            'whale_intent': whale_intent,
            'confidence': confidence,
            'intent_scores': intent_scores,
            'signals': signals
        }

    def _interpret_funding_oi(self, funding_rate: float, oi_change: float, ls_ratio: float) -> str:
        """解释资金费率和持仓变化的含义"""
        if funding_rate > 0.001 and oi_change > 0.05:
            return "BULLISH_MOMENTUM"  # 多头动能强劲
        elif funding_rate < -0.001 and oi_change > 0.05:
            return "SHORT_SQUEEZE_SETUP"  # 可能的空头挤压
        elif funding_rate > 0.001 and oi_change < -0.05:
            return "LONG_LIQUIDATION"  # 多头平仓
        elif funding_rate < -0.001 and oi_change < -0.05:
            return "SHORT_COVERING"  # 空头回补
        else:
            return "NEUTRAL"

    def _calculate_order_book_imbalance(self, bids: List[Tuple[float, float]],
                                       asks: List[Tuple[float, float]]) -> float:
        """计算订单簿失衡度"""
        if not bids or not asks:
            return 0.0

        # 计算不同深度的失衡度
        depths = [5, 10, 20]
        imbalances = []

        for depth in depths:
            bid_sum = sum(qty for _, qty in bids[:depth])
            ask_sum = sum(qty for _, qty in asks[:depth])

            if bid_sum + ask_sum > 0:
                imbalance = (bid_sum - ask_sum) / (bid_sum + ask_sum)
                imbalances.append(imbalance)

        # 加权平均，近端权重更高
        weights = [0.5, 0.3, 0.2]
        weighted_imbalance = sum(w * i for w, i in zip(weights, imbalances))

        return weighted_imbalance

    def _analyze_order_distribution(self, orders: List[Tuple[float, float]]) -> Dict[str, float]:
        """分析订单分布特征"""
        if not orders:
            return {}

        quantities = [qty for _, qty in orders[:50]]

        return {
            'avg_size': np.mean(quantities),
            'median_size': np.median(quantities),
            'std_dev': np.std(quantities),
            'skewness': self._calculate_skewness(quantities),
            'concentration': max(quantities) / sum(quantities) if sum(quantities) > 0 else 0
        }

    def _calculate_skewness(self, data: List[float]) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0

        return np.mean(((data - mean) / std) ** 3)

    # ========== 日志输出方法 ==========

    def _log_order_book_insights(self, analysis: Dict[str, Any]):
        """详细记录订单簿分析结果"""
        print_colored("    💹 订单簿洞察:", Colors.CYAN)

        # 买卖压力
        pressure = analysis['pressure_ratio']
        pressure_color = Colors.GREEN if pressure > 1.2 else Colors.RED if pressure < 0.8 else Colors.YELLOW
        print_colored(f"      • 买卖压力比: {pressure:.2f}", pressure_color)
        print_colored(f"      • 买单量: {analysis['bid_volume_20']:,.0f}", Colors.INFO)
        print_colored(f"      • 卖单量: {analysis['ask_volume_20']:,.0f}", Colors.INFO)

        # 订单簿失衡
        imbalance = analysis['imbalance']
        imb_color = Colors.GREEN if imbalance > 0.1 else Colors.RED if imbalance < -0.1 else Colors.YELLOW
        print_colored(f"      • 订单簿失衡度: {imbalance:.2%}", imb_color)

        # 冰山单
        if analysis['iceberg_orders']['buy'] or analysis['iceberg_orders']['sell']:
            print_colored("      • 🧊 检测到冰山单:", Colors.WARNING)
            for iceberg in analysis['iceberg_orders']['buy'][:2]:
                print_colored(f"        - 买方 @ ${iceberg['price']:.4f} "
                            f"(可见: {iceberg['visible_qty']:,.0f}, "
                            f"预估总量: {iceberg['estimated_total']:,.0f})", Colors.GREEN)
            for iceberg in analysis['iceberg_orders']['sell'][:2]:
                print_colored(f"        - 卖方 @ ${iceberg['price']:.4f} "
                            f"(可见: {iceberg['visible_qty']:,.0f}, "
                            f"预估总量: {iceberg['estimated_total']:,.0f})", Colors.RED)

        # 订单墙
        if analysis['support_walls'] or analysis['resistance_walls']:
            print_colored("      • 🧱 订单墙:", Colors.WARNING)
            for wall in analysis['support_walls'][:1]:
                print_colored(f"        - 支撑墙 @ ${wall['price']:.4f} "
                            f"(数量: {wall['quantity']:,.0f}, 强度: {wall['strength']:.1f}x)", Colors.GREEN)
            for wall in analysis['resistance_walls'][:1]:
                print_colored(f"        - 阻力墙 @ ${wall['price']:.4f} "
                            f"(数量: {wall['quantity']:,.0f}, 强度: {wall['strength']:.1f}x)", Colors.RED)

    def _log_spot_flow_insights(self, analysis: Dict[str, Any]):
        """详细记录现货大单分析结果"""
        if not analysis or analysis.get('whale_trades_count', 0) == 0:
            print_colored("    🐋 现货大单: 无显著活动", Colors.GRAY)
            return

        print_colored("    🐋 现货大单分析:", Colors.CYAN)

        # 净流向
        net_flow = analysis['whale_net_flow']
        flow_color = Colors.GREEN if net_flow > 0 else Colors.RED
        print_colored(f"      • 净流向: {flow_color}${abs(net_flow):,.0f}{Colors.RESET}", Colors.INFO)
        print_colored(f"      • 买入量: ${analysis['whale_buy_volume']:,.0f}", Colors.GREEN)
        print_colored(f"      • 卖出量: ${analysis['whale_sell_volume']:,.0f}", Colors.RED)
        print_colored(f"      • 大单数量: {analysis['whale_trades_count']} "
                     f"({analysis['whale_ratio']:.1%})", Colors.INFO)

        # 最大的几笔交易
        if 'largest_trades' in analysis and analysis['largest_trades']:
            print_colored("      • 最大交易:", Colors.INFO)
            for trade in analysis['largest_trades'][:3]:
                side_color = Colors.GREEN if trade['is_buy'] else Colors.RED
                print_colored(f"        - {side_color}{'买入' if trade['is_buy'] else '卖出'}{Colors.RESET} "
                            f"{trade['qty']:.2f} @ ${trade['price']:.4f} "
                            f"(${trade['quoteQty']:,.0f})", Colors.INFO)

    def _log_funding_insights(self, analysis: Dict[str, Any]):
        """详细记录资金费率分析结果"""
        if not analysis:
            return

        print_colored("    💰 资金面分析:", Colors.CYAN)

        # 资金费率
        funding = analysis['funding_rate']
        funding_color = Colors.RED if abs(funding) > 0.001 else Colors.YELLOW if abs(funding) > 0.0005 else Colors.GREEN
        print_colored(f"      • 资金费率: {funding:.4%} ({analysis['funding_direction']})", funding_color)

        # 持仓量变化
        oi_change = analysis['oi_change_1h']
        oi_color = Colors.GREEN if abs(oi_change) > 0.05 else Colors.YELLOW if abs(oi_change) > 0.02 else Colors.GRAY
        print_colored(f"      • 持仓变化(1h): {oi_change:+.1%}", oi_color)
        print_colored(f"      • 当前持仓: {analysis['open_interest']:,.0f}", Colors.INFO)

        # 多空比
        ls_ratio = analysis['long_short_ratio']
        ls_color = Colors.GREEN if ls_ratio > 1.2 else Colors.RED if ls_ratio < 0.8 else Colors.YELLOW
        print_colored(f"      • 多空比: {ls_ratio:.2f}", ls_color)

        # 市场情绪解读
        sentiment = analysis['market_sentiment']
        sentiment_map = {
            'BULLISH_MOMENTUM': ('多头势头强劲 🚀', Colors.GREEN),
            'SHORT_SQUEEZE_SETUP': ('潜在轧空机会 ⚡', Colors.YELLOW),
            'LONG_LIQUIDATION': ('多头清算中 📉', Colors.RED),
            'SHORT_COVERING': ('空头回补中 📈', Colors.GREEN),
            'NEUTRAL': ('市场情绪中性 ➖', Colors.GRAY)
        }
        sent_text, sent_color = sentiment_map.get(sentiment, ('未知', Colors.GRAY))
        print_colored(f"      • 市场情绪: {sent_text}", sent_color)

    def _log_technical_insights(self, analysis: Dict[str, Any]):
        """详细记录技术指标分析结果"""
        if not analysis:
            return

        print_colored("    📈 技术指标:", Colors.CYAN)

        # RSI
        rsi = analysis['rsi']
        rsi_signal = analysis['rsi_signal']
        rsi_color = Colors.RED if rsi > 70 else Colors.GREEN if rsi < 30 else Colors.YELLOW
        print_colored(f"      • RSI(14): {rsi:.1f} ({rsi_signal})", rsi_color)

        # MACD
        macd_cross = analysis['macd_cross']
        if macd_cross != 'NONE':
            cross_color = Colors.GREEN if macd_cross == 'BULLISH' else Colors.RED
            print_colored(f"      • MACD: {macd_cross} CROSS", cross_color)

        # 布林带位置
        bb_pos = analysis['bb_position']
        bb_color = Colors.RED if bb_pos > 0.9 else Colors.GREEN if bb_pos < 0.1 else Colors.YELLOW
        print_colored(f"      • 布林带位置: {bb_pos:.1%}", bb_color)

        # 成交量
        if analysis['volume_surge']:
            print_colored(f"      • ⚡ 成交量激增 (比率: {analysis.get('volume_ratio', 0):.1f}x)", Colors.WARNING)

        # 技术评分
        tech_score = analysis['technical_score']
        score_color = Colors.GREEN if tech_score > 1 else Colors.RED if tech_score < -1 else Colors.YELLOW
        print_colored(f"      • 技术评分: {tech_score:.1f}", score_color)

    def _log_final_verdict(self, analysis: Dict[str, Any]):
        """输出最终判断结果"""
        print_colored("\n    🎯 综合判断:", Colors.CYAN + Colors.BOLD)

        # 庄家意图
        intent = analysis['whale_intent']
        confidence = analysis['confidence']

        intent_map = {
            'ACCUMULATION': ('吸筹建仓', Colors.GREEN),
            'DISTRIBUTION': ('派发出货', Colors.RED),
            'MANIPULATION_UP': ('拉升操纵', Colors.YELLOW),
            'MANIPULATION_DOWN': ('打压操纵', Colors.YELLOW),
            'NEUTRAL': ('意图不明', Colors.GRAY)
        }

        intent_text, intent_color = intent_map.get(intent, ('未知', Colors.GRAY))
        print_colored(f"      • 庄家意图: {intent_text}", intent_color + Colors.BOLD)
        print_colored(f"      • 置信度: {confidence:.1%}", Colors.INFO)

        # 交易建议
        recommendation = analysis['recommendation']
        rec_map = {
            'BUY': ('建议买入 🟢', Colors.GREEN),
            'SELL': ('建议卖出 🔴', Colors.RED),
            'BUY_CAUTIOUS': ('谨慎做多 ⚠️', Colors.YELLOW),
            'SELL_CAUTIOUS': ('谨慎做空 ⚠️', Colors.YELLOW),
            'HOLD': ('观望等待 ⏸️', Colors.GRAY)
        }

        rec_text, rec_color = rec_map.get(recommendation, ('观望', Colors.GRAY))
        print_colored(f"      • 交易建议: {rec_text}", rec_color + Colors.BOLD)

        # 关键信号
        if analysis.get('signals'):
            print_colored("      • 关键信号:", Colors.INFO)
            for signal in analysis['signals'][:5]:  # 最多显示5个
                print_colored(f"        - {signal}", Colors.INFO)

        # 风险提示
        if analysis.get('risk_factors'):
            print_colored("      • ⚠️ 风险因素:", Colors.WARNING)
            for risk in analysis['risk_factors'][:3]:
                print_colored(f"        - {risk}", Colors.WARNING)