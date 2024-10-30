import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

class InteractiveRectangleDrawer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.rect_selector = RectangleSelector(self.ax, self.on_select,
                                               drawtype='box', useblit=True,
                                               button=[1],  # 左クリックで選択
                                               minspanx=5, minspany=5,
                                               spancoords='pixels',
                                               interactive=True)
        self.rectangles = []
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_select(self, eclick, erelease):
        """矩形選択イベント時に呼び出されるコールバック"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.rectangles.append(((x1, y1), (x2, y2)))
        print(f"矩形が選択されました: {(x1, y1)} → {(x2, y2)}")

    def on_key_press(self, event):
        """キー押下イベント時に呼び出されるコールバック"""
        if event.key == 'q':
            plt.close(self.fig)

    def show(self):
        plt.title("Draw rectangle by drag and drop. Press 'q' when you want to quit.")
        plt.show()

# インタラクティブに矩形を描画する
drawer = InteractiveRectangleDrawer()
drawer.show()
