"""
GUI界面 - PyQt5实现
"""
import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QGroupBox,
    QFrame, QSplitter, QStatusBar, QProgressBar, QCheckBox,
    QFileDialog, QMessageBox, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import deque
import time
import torch

from game import Game2048
from model import Game2048Transformer
from trainer import PPOTrainer, RolloutBuffer, Transition


# 砖块颜色配置
TILE_COLORS = {
    0: '#cdc1b4',
    2: '#eee4da',
    4: '#ede0c8',
    8: '#f2b179',
    16: '#f59563',
    32: '#f67c5f',
    64: '#f65e3b',
    128: '#edcf72',
    256: '#edcc61',
    512: '#edc850',
    1024: '#edc53f',
    2048: '#edc22e',
    4096: '#3c3a32',
    8192: '#3c3a32',
}

TILE_TEXT_COLORS = {
    2: '#776e65',
    4: '#776e65',
}


class GameBoardWidget(QWidget):
    """2048游戏面板组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.cell_size = 80
        self.padding = 5
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setFixedSize(
            self.cell_size * 4 + self.padding * 5,
            self.cell_size * 4 + self.padding * 5
        )
    
    def set_board(self, board: np.ndarray):
        """设置棋盘状态"""
        self.board = board.copy()
        self.update()
    
    def paintEvent(self, event):
        """绘制棋盘"""
        from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 背景
        painter.fillRect(self.rect(), QColor('#bbada0'))
        
        # 绘制每个格子
        for i in range(4):
            for j in range(4):
                x = self.padding + j * (self.cell_size + self.padding)
                y = self.padding + i * (self.cell_size + self.padding)
                value = self.board[i, j]
                
                # 格子颜色
                color = TILE_COLORS.get(value, '#3c3a32')
                painter.fillRect(x, y, self.cell_size, self.cell_size, QColor(color))
                
                # 数字
                if value > 0:
                    # 文字颜色
                    text_color = TILE_TEXT_COLORS.get(value, '#f9f6f2')
                    painter.setPen(QColor(text_color))
                    
                    # 字体大小根据数字位数调整
                    if value < 100:
                        font_size = 32
                    elif value < 1000:
                        font_size = 28
                    else:
                        font_size = 22
                    
                    font = QFont('Arial', font_size, QFont.Bold)
                    painter.setFont(font)
                    
                    # 居中绘制
                    text = str(int(value))
                    painter.drawText(x, y, self.cell_size, self.cell_size,
                                    Qt.AlignCenter, text)


class ScoreWidget(QWidget):
    """分数显示组件"""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.value = 0
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                background-color: #bbada0;
                color: #eee4da;
                font-size: 12px;
                font-weight: bold;
                border-radius: 3px;
                padding: 5px;
            }
        """)
        layout.addWidget(self.title_label)
        
        # 数值
        self.value_label = QLabel('0')
        self.value_label.setAlignment(Qt.AlignCenter)
        self.value_label.setStyleSheet("""
            QLabel {
                background-color: #8f7a66;
                color: white;
                font-size: 20px;
                font-weight: bold;
                border-radius: 3px;
                padding: 10px;
                min-width: 80px;
            }
        """)
        layout.addWidget(self.value_label)
    
    def set_value(self, value):
        """设置分数值"""
        self.value = value
        if isinstance(value, float):
            self.value_label.setText(f'{value:.1f}')
        else:
            self.value_label.setText(str(int(value)))


class PlotCanvas(FigureCanvas):
    """matplotlib绑定的画布"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.fig.patch.set_facecolor('#faf8ef')
        self.axes.set_facecolor('#faf8ef')
        
        # 设置中文字体
        self.axes.set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        self.fig.tight_layout()
    
    def plot_training_scores(self, scores, title="Training Scores"):
        """绘制训练曲线 - 只显示累积分数"""
        self.axes.clear()
        
        if scores:
            x = range(1, len(scores) + 1)
            self.axes.plot(x, scores, '#1f77b4', linewidth=1.5, alpha=0.8)
            
            # 添加移动平均线
            if len(scores) >= 10:
                window = min(50, len(scores) // 5)
                if window >= 5:
                    ma = np.convolve(scores, np.ones(window)/window, mode='valid')
                    ma_x = range(window, len(scores) + 1)
                    self.axes.plot(ma_x, ma, '#d62728', linewidth=2, alpha=0.8, label=f'MA({window})')
        
        self.axes.set_xlabel('Games', fontsize=10)
        self.axes.set_ylabel('Score', fontsize=10)
        self.axes.set_title(title, fontsize=12)
        self.axes.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_demo_scores(self, scores, title="Situational Score"):
        """绘制演示曲线 - 只显示局面分数"""
        self.axes.clear()
        
        if scores:
            x = range(1, len(scores) + 1)
            self.axes.plot(x, scores, '#2ca02c', linewidth=1.5, alpha=0.8)
        
        self.axes.set_xlabel('Steps', fontsize=10)
        self.axes.set_ylabel('Situational Score', fontsize=10)
        self.axes.set_title(title, fontsize=12)
        self.axes.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.draw()


class SimpleTrainingThread(QThread):
    """简化训练线程"""
    
    game_end_signal = pyqtSignal(dict)
    progress_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    
    def __init__(self, model, trainer, num_games=1000):
        super().__init__()
        self.model = model
        self.trainer = trainer
        self.num_games = num_games
        self.running = True
        self.device = "cpu"
    
    def run(self):
        """运行训练"""
        scores = []
        max_tiles = []
        best_score = 0
        start_time = time.time()
        
        for game_idx in range(self.num_games):
            if not self.running:
                break
            
            game = Game2048()
            game.reset()
            buffer = RolloutBuffer(capacity=10000)
            
            while not game.game_over and self.running:
                state = game.get_state()
                scores_feat = np.array([
                    min(game.accumulated_score / 50000, 1.0),
                    min(game.situational_score / 200, 1.0)
                ], dtype=np.float32)
                valid = game.get_valid_actions()
                
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                scores_t = torch.FloatTensor(scores_feat).unsqueeze(0).to(self.device)
                valid_t = torch.BoolTensor(valid).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, log_prob, value = self.model.get_action(
                        state_t, scores_t, valid_t, deterministic=False
                    )
                
                old_state = state.copy()
                old_scores = scores_feat.copy()
                next_state, reward, moved, done = game.move(action)
                
                transition = Transition(
                    state=old_state,
                    scores=old_scores,
                    action=action,
                    reward=reward,
                    next_state=next_state.copy(),
                    next_scores=np.array([
                        min(game.accumulated_score / 50000, 1.0),
                        min(game.situational_score / 200, 1.0)
                    ], dtype=np.float32),
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    valid_actions=valid
                )
                buffer.push(transition)
                
                if len(buffer) >= 64:
                    self.trainer.update(buffer)
                    buffer.clear()
            
            scores.append(game.accumulated_score)
            max_tiles.append(game.get_max_tile())
            
            if game.accumulated_score > best_score:
                best_score = game.accumulated_score
            
            # 发送游戏结束信号
            elapsed = time.time() - start_time
            self.game_end_signal.emit({
                'score': game.accumulated_score,
                'max_tile': game.get_max_tile(),
                'game_idx': game_idx + 1,
                'best_score': best_score,
                'elapsed': elapsed,
                'avg_score': np.mean(scores[-100:]) if scores else 0
            })
        
        self.finished_signal.emit()
    
    def stop(self):
        """停止训练"""
        self.running = False


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('2048 AI Trainer')
        self.setMinimumSize(1000, 700)
        
        # 初始化模型和训练器
        self.device = 'cpu'
        self.model = Game2048Transformer()
        self.trainer = PPOTrainer(self.model, lr=3e-4, device=self.device)
        
        # 游戏实例（用于演示）
        self.game = Game2048()
        
        # 训练状态
        self.is_training = False
        self.training_thread = None
        
        # 统计数据
        self.training_scores = []
        self.demo_situational_scores = []
        
        # 演示模式状态
        self.ai_mode = False
        self.auto_step = False
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.ai_step)
        
        # 当前模型路径
        self.current_model_path = None
        
        self.init_ui()
        self.update_display()
    
    def init_ui(self):
        """初始化UI"""
        # 主窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #faf8ef;
            }
            QLabel {
                color: #776e65;
            }
            QPushButton {
                background-color: #8f7a66;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #9f8b77;
            }
            QPushButton:pressed {
                background-color: #7f6a57;
            }
            QPushButton:disabled {
                background-color: #ccc;
                color: #999;
            }
            QComboBox {
                background-color: #8f7a66;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QGroupBox {
                border: 2px solid #bbada0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #776e65;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QSpinBox {
                background-color: white;
                border: 1px solid #bbada0;
                border-radius: 3px;
                padding: 5px;
                min-width: 80px;
            }
        """)
        
        # 中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧面板（游戏区）
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel)
        
        # 右侧面板（统计和控制）
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel)
        
        # 设置拉伸比例
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 2)
        
        # 状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Ready')
    
    def create_left_panel(self) -> QWidget:
        """创建左侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignCenter)
        
        # 模式选择
        mode_layout = QHBoxLayout()
        mode_label = QLabel('Mode:')
        mode_label.setFont(QFont('Arial', 12, QFont.Bold))
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Training Mode', 'Demo Mode'])
        self.mode_combo.currentIndexChanged.connect(self.switch_mode)
        
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # 分数显示
        scores_layout = QHBoxLayout()
        self.accumulated_score_widget = ScoreWidget('Score')
        self.situational_score_widget = ScoreWidget('Situational')
        self.max_tile_widget = ScoreWidget('Max Tile')
        
        scores_layout.addWidget(self.accumulated_score_widget)
        scores_layout.addWidget(self.situational_score_widget)
        scores_layout.addWidget(self.max_tile_widget)
        layout.addLayout(scores_layout)
        
        # 游戏面板
        self.game_board = GameBoardWidget()
        layout.addWidget(self.game_board, alignment=Qt.AlignCenter)
        
        # 演示模式控制
        self.demo_controls = QWidget()
        demo_layout = QHBoxLayout(self.demo_controls)
        
        self.ai_btn = QPushButton('AI Mode')
        self.ai_btn.clicked.connect(self.toggle_ai_mode)
        
        self.step_btn = QPushButton('Step')
        self.step_btn.clicked.connect(self.ai_step)
        
        self.auto_btn = QPushButton('Auto')
        self.auto_btn.clicked.connect(self.toggle_auto)
        
        self.reset_btn = QPushButton('Reset')
        self.reset_btn.clicked.connect(self.reset_game)
        
        demo_layout.addWidget(self.ai_btn)
        demo_layout.addWidget(self.step_btn)
        demo_layout.addWidget(self.auto_btn)
        demo_layout.addWidget(self.reset_btn)
        
        self.demo_controls.setVisible(False)
        layout.addWidget(self.demo_controls)
        
        layout.addStretch()
        return panel
    
    def create_right_panel(self) -> QWidget:
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 训练控制
        control_group = QGroupBox('Training Control')
        control_layout = QVBoxLayout(control_group)
        
        # 训练局数设置
        games_layout = QHBoxLayout()
        games_label = QLabel('Games:')
        self.games_spinbox = QSpinBox()
        self.games_spinbox.setRange(10, 100000)
        self.games_spinbox.setValue(500)
        games_layout.addWidget(games_label)
        games_layout.addWidget(self.games_spinbox)
        games_layout.addStretch()
        control_layout.addLayout(games_layout)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton('Start Training')
        self.start_btn.clicked.connect(self.start_training)
        
        self.stop_btn = QPushButton('Stop Training')
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        control_layout.addLayout(btn_layout)
        
        # 模型文件操作
        model_layout = QHBoxLayout()
        self.load_btn = QPushButton('Load Model')
        self.load_btn.clicked.connect(self.load_model)
        
        self.save_btn = QPushButton('Save Model')
        self.save_btn.clicked.connect(self.save_model)
        
        model_layout.addWidget(self.load_btn)
        model_layout.addWidget(self.save_btn)
        control_layout.addLayout(model_layout)
        
        # 训练参数显示
        param_layout = QGridLayout()
        
        self.games_label = QLabel('Games: 0')
        self.avg_score_label = QLabel('Avg Score: 0')
        self.best_score_label = QLabel('Best Score: 0')
        self.speed_label = QLabel('Speed: 0 games/s')
        
        param_layout.addWidget(self.games_label, 0, 0)
        param_layout.addWidget(self.avg_score_label, 0, 1)
        param_layout.addWidget(self.best_score_label, 1, 0)
        param_layout.addWidget(self.speed_label, 1, 1)
        
        control_layout.addLayout(param_layout)
        layout.addWidget(control_group)
        
        # 分数曲线
        plot_group = QGroupBox('Score Chart')
        plot_layout = QVBoxLayout(plot_group)
        
        self.plot_canvas = PlotCanvas(self, width=6, height=4, dpi=100)
        plot_layout.addWidget(self.plot_canvas)
        
        layout.addWidget(plot_group)
        
        # 实时统计
        stats_group = QGroupBox('Training Stats')
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QLabel('Waiting for training...')
        self.stats_text.setStyleSheet('font-family: monospace;')
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        
        return panel
    
    def switch_mode(self, index):
        """切换模式"""
        if index == 0:  # 训练模式
            self.demo_controls.setVisible(False)
            self.demo_situational_scores = []
            self.plot_canvas.plot_training_scores(self.training_scores)
        else:  # 演示模式
            self.demo_controls.setVisible(True)
            self.reset_game()
            self.demo_situational_scores = []
            self.plot_canvas.plot_demo_scores([])
    
    def keyPressEvent(self, event):
        """键盘事件"""
        if self.mode_combo.currentIndex() == 1 and not self.ai_mode:
            # 演示模式且非AI托管
            key_map = {
                Qt.Key_Up: 0,
                Qt.Key_Down: 1,
                Qt.Key_Left: 2,
                Qt.Key_Right: 3,
            }
            
            if event.key() in key_map:
                direction = key_map[event.key()]
                self.game.move(direction)
                self.demo_situational_scores.append(self.game.situational_score)
                self.update_display()
                self.plot_canvas.plot_demo_scores(self.demo_situational_scores)
                
                if self.game.game_over:
                    self.statusBar.showMessage('Game Over!')
    
    def toggle_ai_mode(self):
        """切换AI托管模式"""
        self.ai_mode = not self.ai_mode
        if self.ai_mode:
            self.ai_btn.setText('Manual')
            self.step_btn.setEnabled(False)
            self.auto_btn.setEnabled(True)
        else:
            self.ai_btn.setText('AI Mode')
            self.step_btn.setEnabled(True)
            self.auto_btn.setEnabled(False)
            self.auto_timer.stop()
            self.auto_btn.setText('Auto')
    
    def ai_step(self):
        """AI单步执行"""
        if self.game.game_over:
            self.reset_game()
            return
        
        state = self.game.get_state()
        scores = np.array([
            min(self.game.accumulated_score / 50000, 1.0),
            min(self.game.situational_score / 200, 1.0)
        ], dtype=np.float32)
        valid_actions = self.game.get_valid_actions()
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        scores_t = torch.FloatTensor(scores).unsqueeze(0)
        valid_t = torch.BoolTensor(valid_actions).unsqueeze(0)
        
        action, _, _ = self.model.get_action(state_t, scores_t, valid_t, deterministic=True)
        
        self.game.move(action)
        self.demo_situational_scores.append(self.game.situational_score)
        self.update_display()
        self.plot_canvas.plot_demo_scores(self.demo_situational_scores)
        
        if self.game.game_over:
            self.statusBar.showMessage(f'Game Over! Final Score: {self.game.accumulated_score}')
            if self.auto_timer.isActive():
                self.auto_timer.stop()
                self.auto_btn.setText('Auto')
    
    def toggle_auto(self):
        """切换自动执行"""
        if self.auto_timer.isActive():
            self.auto_timer.stop()
            self.auto_btn.setText('Auto')
        else:
            self.auto_timer.start(100)  # 100ms间隔
            self.auto_btn.setText('Stop')
    
    def reset_game(self):
        """重置游戏"""
        self.game.reset()
        self.demo_situational_scores = [self.game.situational_score]
        self.update_display()
        self.plot_canvas.plot_demo_scores(self.demo_situational_scores)
        self.statusBar.showMessage('Game Reset')
    
    def update_display(self):
        """更新显示"""
        self.game_board.set_board(self.game.board)
        self.accumulated_score_widget.set_value(self.game.accumulated_score)
        self.situational_score_widget.set_value(self.game.situational_score)
        self.max_tile_widget.set_value(self.game.get_max_tile())
    
    def start_training(self):
        """开始训练"""
        self.is_training = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.games_spinbox.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        # 重置统计
        self.training_scores = []
        
        # 创建训练线程
        num_games = self.games_spinbox.value()
        self.training_thread = SimpleTrainingThread(
            self.model, self.trainer, num_games
        )
        self.training_thread.game_end_signal.connect(self.on_game_end)
        self.training_thread.finished_signal.connect(self.on_training_finished)
        self.training_thread.start()
        
        self.statusBar.showMessage('Training started...')
    
    def stop_training(self):
        """停止训练"""
        if self.training_thread:
            self.training_thread.stop()
            self.training_thread.wait()
            self.training_thread = None
        
        self.on_training_finished()
    
    def on_game_end(self, stats):
        """游戏结束回调"""
        self.training_scores.append(stats['score'])
        
        # 更新统计显示
        games = stats['game_idx']
        avg_score = stats['avg_score']
        best_score = stats['best_score']
        elapsed = stats['elapsed']
        speed = games / elapsed if elapsed > 0 else 0
        
        self.games_label.setText(f'Games: {games}')
        self.avg_score_label.setText(f'Avg Score: {avg_score:.0f}')
        self.best_score_label.setText(f'Best Score: {best_score}')
        self.speed_label.setText(f'Speed: {speed:.2f} games/s')
        
        # 更新曲线
        if games % 5 == 0:
            self.plot_canvas.plot_training_scores(self.training_scores)
        
        # 更新统计文本
        self.stats_text.setText(
            f"Games: {games}\n"
            f"Avg Score: {avg_score:.0f}\n"
            f"Best Score: {best_score}\n"
            f"Last Score: {stats['score']}"
        )
    
    def on_training_finished(self):
        """训练完成回调"""
        self.is_training = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.games_spinbox.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # 自动保存模型
        save_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'model.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_scores': self.training_scores,
            'best_score': max(self.training_scores) if self.training_scores else 0
        }, save_path)
        
        self.statusBar.showMessage(f'Training finished! Model saved to {save_path}')
    
    def load_model(self):
        """加载模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Model', 
            os.path.join(os.path.dirname(__file__), 'checkpoints'),
            'PyTorch Model (*.pt);;All Files (*)'
        )
        
        if file_path:
            try:
                checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.current_model_path = file_path
                
                if 'training_scores' in checkpoint:
                    self.training_scores = checkpoint['training_scores']
                    self.plot_canvas.plot_training_scores(self.training_scores)
                
                self.statusBar.showMessage(f'Model loaded: {os.path.basename(file_path)}')
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to load model:\n{str(e)}')
    
    def save_model(self):
        """保存模型"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Model',
            os.path.join(os.path.dirname(__file__), 'checkpoints', 'model.pt'),
            'PyTorch Model (*.pt);;All Files (*)'
        )
        
        if file_path:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'training_scores': self.training_scores,
                    'best_score': max(self.training_scores) if self.training_scores else 0
                }, file_path)
                self.current_model_path = file_path
                self.statusBar.showMessage(f'Model saved: {os.path.basename(file_path)}')
            except Exception as e:
                QMessageBox.warning(self, 'Error', f'Failed to save model:\n{str(e)}')


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置字体
    font = QFont('Arial', 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()