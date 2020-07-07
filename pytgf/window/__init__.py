"""
Package containing the window related classes. Only import it if you actually need to display something on the screen,
for other purpose, you can use window-less rendering which will be more efficient.
"""

from pytgf.window.window import Window, WindowRenderLoop, AssetManager

from pytgf.logic.physics import World
from pytgf.graphics.graphics import Camera
from pytgf.graphics import Game

import numpy


class WindowedGame(Game):
    """
    A standard game object.

    This framework is designed to create a complete game (logic and rendering). This is supposed to be used in windowed
    mode, see Game for windowless (standalone) mode.

    Attributes
    ----------
    resources: AssetManager
        The main resource manager used for the rendering.
    world: World
        The game world.
    input_handler: Window
        The main input handler.
    camera: Camera
        The camera used for the rendering.
    gui: GUIManager
        The main GUI container of the game.
    history: list of Event
        The list of fired events.
    history: list of Event
        The list of fired events.

    Methods
    -------
    update(tick)
        Updates the world and fires the input events.
    render(frame)
        Renders the world and the GUI.
    run()
        Runs the game logic loop.
    reset()
        Resets the game.
    close()
        Closes the game.
    change_world(tiles, background, logic_area, logic_tile, logic_entity, entity_per_thread, node_capacity, max_depth)
        Creates a new world.
    fire_event(event)
        Handles a new fired Event.
    register_event_handler(event_type, handler)
        Registers a new event handler.
    register_collision_event_handler(handler)
        Registers a new CollisionEvent handler.
    register_collision_tile_event_handler(handler)
        Registers a new CollisionWithTileEvent handler.
    register_collision_entity_event_handler(handler)
        Registers a new CollisionWithEntityEvent handler.
    register_input_event_handler(handler)
        Registers a new InputEvent handler.
    register_key_event_handler(handler)
        Registers a new KeyEvent handler.
    register_key_pressed_event_handler(handler)
        Registers a new KeyPressedEvent handler.
    register_key_released_event_handler(handler)
        Registers a new KeyReleasedEvent handler.
    register_key_typed_event_handler(handler)
        Registers a new KeyTypedEvent handler.
    register_key_held_event_handler(handler)
        Registers a new KeyHeldEvent handler.
    register_mouse_event_handler(handler)
        Registers a new MouseEvent handler.
    register_mouse_moved_event_handler(handler)
        Registers a new MouseMovedEvent handler.
    register_mouse_button_event_handler(handler)
        Registers a new MouseButtonEvent handler.
    register_mouse_button_pressed_event_handler(handler)
        Registers a new MouseButtonPressedEvent handler.
    register_mouse_button_released_event_handler(handler)
        Registers a new MouseButtonReleasedEvent handler.
    register_mouse_button_clicked_event_handler(handler)
        Registers a new MouseButtonClickedEvent handler.
    register_mouse_dragged_event_handler(handler)
        Registers a new MouseDraggedEvent handler.
    register_gui_event_handler(handler)
        Registers a new GUIEvent handler.
    register_gui_focused_event_handler(handler)
        Registers a new GUIFocusedEvent handler.
    register_gui_unfocused_event_handler(handler)
        Registers a new GUIUnfocusedEvent handler.
    """

    def __init__(self, tile_size: int, viewport: [tuple, numpy.ndarray], title: str, scale: float = 1.0,
                 width: int = 800, height: int = 600, full_screen: bool = True, hide_cursor: bool = False,
                 glsl_version: int = 330, shader_world: tuple = None, shader_sprite: tuple = None,
                 tick_per_second: float = 60, frame_per_second: float = 60, multi_threading: bool = True,
                 default_tile_collision_handler: bool = True, default_entity_collision_handler: bool = True,
                 default_gui_handler: bool = True):
        """
        Initializes the WindowedGame.

        Parameters
        ----------
        tile_size: int
            The size of the tiles expressed in distance units.
        viewport: [tuple, numpy.ndarray]
            The size of the field of view.
        title: str
            The name of the window.
        scale: float, optional
            The rendering scale factor.
        width: int, optional
            The width of the window if not created in full screen.
        height: int, optional
            The height of the window if not created in full screen.
        full_screen: bool, optional
            Creates the a full screen window if set to True.
        hide_cursor: bool, optional
            Hides the cursor in the window area if set to True.
        glsl_version: int, optional
            The required GLSL version.
        shader_world: tuple of strings, optional
            The couple of source code for the fragment and vertex shaders used for the rendering of the world.
        shader_sprite: tuple of strings, optional
            The couple of source code for the fragment and vertex shaders used for the rendering of sprites.
        tick_per_second: float, optional
            The tick rate of the loop. It correspond to the number of times the logic will be performed per second.
        frame_per_second: float, optional
            The frame rate of the loop. It correspond to the number of times the frame will be rendered per second.
        multi_threading: bool, optional
            Enables the multi-threading mode if set to True.
        default_tile_collision_handler: bool, optional
            Registers the default tile collision event handler if set to True.
        default_entity_collision_handler: bool, optional
            Registers the default entity collision event handler if set to True.
        default_gui_handler: bool: optional
            Registers the default GUI event handler if set to True.
        """

        local_window = Window(
            title, self, width=width, height=height, full_screen=full_screen, hide_cursor=hide_cursor
        )

        self.resources = AssetManager(
            tile_size, scale=scale, standalone=False, glsl_version=glsl_version, shader_world=shader_world,
            shader_sprite=shader_sprite, sampling_rate=frame_per_second
        )

        self.input_handler = local_window

        self._loop = WindowRenderLoop(
            tick_per_second, frame_per_second, self.update, self.render, self.close, self.input_handler
        )

        Game.__init__(
            self, tile_size, viewport, scale=scale, standalone=False, glsl_version=glsl_version,
            shader_world=shader_world, shader_sprite=shader_sprite, tick_per_second=tick_per_second,
            frame_per_second=frame_per_second, multi_threading=multi_threading, default_gui_handler=default_gui_handler,
            default_tile_collision_handler=default_tile_collision_handler,
            default_entity_collision_handler=default_entity_collision_handler
        )

    def render(self, frame: int) -> None:
        """
        Renders the world and the GUI.

        Parameters
        ----------
        frame: int
            The current frame.
        """

        super().render(frame)

        self.resources.update()

    def close(self) -> None:
        """
        Closes the game (this function should clean up the memory).
        """

        super().close()

        self.resources.clear()

        self.input_handler.should_close = True
