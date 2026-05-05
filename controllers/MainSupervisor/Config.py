from ConsoleLog import Console


class Config():
    """Holds data for the settings configuration of Erebus
    """

    def __init__(self, config_data: list[str], path: str):
        """Initialises config settings data

        Args:
            config_data (list[str]): List of config settings
            path (str): Path to config.txt file
        """
        # config_data format
        # [0]: Keep controller/robot files
        # [1]: Disable auto LoP
        # [2]: Recording
        # [3]: Automatic camera
        # [4]: Keep remote
        # [5]: Debug enabled
        # [6]: Docker path
        # [7]: Fixed route enabled

        self.path: str = path

        def config_bool(index: int, default: bool = False) -> bool:
            if len(config_data) <= index or config_data[index].strip() == "":
                return default
            return bool(int(config_data[index]))

        self.keep_controller: bool = config_bool(0)
        self.disable_lop: bool = config_bool(1)
        self.recording: bool = config_bool(2)
        self.automatic_camera: bool = config_bool(3)
        self.fixed_route_enabled: bool = config_bool(7, True)
        
        # Keep v23 compatibility
        self.keep_remote: bool = False  
        self.docker_path: str = ""

        if len(config_data) >= 5:
            self.keep_remote = config_bool(4)
            Console.update_debug_mode(config_bool(5))
            self.docker_path = str(config_data[6])
