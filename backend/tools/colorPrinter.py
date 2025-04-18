class ColorPrinter:
    """
    支持打印加粗彩色文本的类，颜色包括：
    - 红色 (31)
    - 绿色 (32)
    - 黄色 (33)
    - 蓝色 (34)
    - 品红 (35)
    - 青色 (36)
    - 白色 (37)
    """

    COLORS = {
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 37,
    }

    @classmethod
    def _colorize(cls, text: str, color_code: int) -> str:
        """内部方法：给文本添加颜色和加粗样式"""
        return f"\033[1;{color_code}m{text}\033[0m"  # 添加 1; 表示加粗

    @classmethod
    def red(cls, text: str) -> None:
        """打印加粗红色文本"""
        print(cls._colorize(text, cls.COLORS['red']))

    @classmethod
    def green(cls, text: str) -> None:
        """打印加粗绿色文本"""
        print(cls._colorize(text, cls.COLORS['green']))

    @classmethod
    def yellow(cls, text: str) -> None:
        """打印加粗黄色文本"""
        print(cls._colorize(text, cls.COLORS['yellow']))

    @classmethod
    def blue(cls, text: str) -> None:
        """打印加粗蓝色文本"""
        print(cls._colorize(text, cls.COLORS['blue']))

    @classmethod
    def magenta(cls, text: str) -> None:
        """打印加粗品红文本"""
        print(cls._colorize(text, cls.COLORS['magenta']))

    @classmethod
    def cyan(cls, text: str) -> None:
        """打印加粗青色文本"""
        print(cls._colorize(text, cls.COLORS['cyan']))

    @classmethod
    def white(cls, text: str) -> None:
        """打印加粗白色文本"""
        print(cls._colorize(text, cls.COLORS['white']))

    @classmethod
    def color_text(cls, text: str, color_name: str) -> str:
        """
        返回加粗的带颜色文本（不直接打印）
        可选颜色: red, green, yellow, blue, magenta, cyan, white
        """
        if color_name not in cls.COLORS:
            raise ValueError(f"未知颜色: {color_name}")
        return cls._colorize(text, cls.COLORS[color_name])