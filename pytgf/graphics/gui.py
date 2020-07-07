"""
Contains every classes related to gui rendering.
"""

from pytgf.logic.event import EventQueue, CancelableEvent, Key, MouseButton, MouseButtonClickedEvent
from pytgf.logic.physics import AxisAlignedBoundingBox, array_format
from pytgf.graphics.graphics import ProjectionMatrix, ResourceManager, ShaderProgram

import numpy


def transform_gui(position: [tuple, numpy.ndarray], projection: ProjectionMatrix) -> numpy.ndarray:
    """
    Transforms the position vector into the GUI coordinate system.

    Transforms the coordinate of a point expressed in the screen referential to the local GUI coordinate system.

    Parameters
    ----------
    position: numpy.ndarray
        The screen relative position vector.
    projection: ProjectionMatrix
        The local projection matrix of the GUI component.

    Returns
    -------
    transformed_position: numpy.ndarray
        The new position vector expressed in the GUI coordinate system.
    """

    transform = projection.inverse_transform(array_format(position, dtype=numpy.float32))

    transform[0] -= projection.matrix[3, 0] / projection.matrix[0, 0]
    transform[1] -= projection.matrix[3, 1] / projection.matrix[1, 1]

    return transform


class GUIFont:
    """
    A standard object used to display text.

    A separate sprite set should be created for the font, in which each animation corresponds to a single character. The
    character map makes the bridge between the indexes of the animations and the actual characters.

    Attributes
    ----------
    sprite_set: str
        The sprite set containing the characters.
    width: float
        The width of the characters.
    height: float
        The height of the characters.

    Methods
    -------
    get_id_animation(char)
        Returns the animation associated to the character.
    """

    def __init__(self, sprite_set: str, char_map: str, width: float, height: float):
        """
        Initializes the GUIFont.

        Parameters
        ----------
        sprite_set: str
            The sprite set containing the characters.
        char_map: str
            The character map of the font.
        width: float
            The width of the characters.
        height: float
            The height of the characters.
        """

        self._char_map = char_map

        self.sprite_set = sprite_set

        self.width = width
        self.height = height

    def get_id_animation(self, char: str) -> int:
        """
        Returns the animation associated to the character.

        Parameters
        ----------
        char: str
            The character to render.

        Returns
        -------
        id_animation: int
            The index of the animation to render.
        """

        for k in range(len(self._char_map)):
            if self._char_map[k] == char:
                return k

        return 0


class GUIBorder:
    """
    A standard object used to display borders.

    A separate sprite set should be created for the border, in which there are the following animations used to render
    each part of the border: bottom left corner, top left corner, bottom right corner, top right corner, left edge,
    right edge, bottom edge, top edge and the background texture (the order have to be respected).

    Methods
    -------
    render(bounds, graphics, projection)
        Renders the border.
    """

    def __init__(self, sprite_set: str, outline_thickness: float):
        """
        Initializes the GUIBorder.

        Parameters
        ----------
        sprite_set: str
            The sprite set containing the border elements.
        outline_thickness: float
            The thickness of the border edges and corners.
        """

        self._sprite_set = sprite_set
        self._outline_thickness = outline_thickness

    def render(self, bounds: AxisAlignedBoundingBox, resources: ResourceManager, projection: ProjectionMatrix) -> None:
        """
        Renders the border.

        Parameters
        ----------
        bounds: AxisAlignedBoundingBox
            The bounding box of which the border is rendered.
        resources: ResourceManager
            The main resource manager used for the rendering.
        projection: ProjectionMatrix
            The local projection matrix of the GUI component.
        """

        resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_PROJECTION, projection.matrix)

        for i in range(2):
            for j in range(2):
                _, texture = resources.sprite_sets[self._sprite_set].get_texture(i * 2 + j, 0)

                texture.bind(0)

                position = ProjectionMatrix().scale(
                    (self._outline_thickness, self._outline_thickness)
                ).translate(
                    (bounds.position[0] + i * bounds.bounds[0], bounds.position[1] + j * bounds.bounds[1])
                )

                resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_POSITION, position.matrix)
                resources.model_sprite.render()

        for i in range(2):
            _, texture = resources.sprite_sets[self._sprite_set].get_texture(4 + i, 0)

            texture.bind(0)

            position = ProjectionMatrix().scale(
                (self._outline_thickness, bounds.bounds[1] - self._outline_thickness)
            ).translate(
                (bounds.position[0] + i * bounds.bounds[0], bounds.position[1] + bounds.bounds[1] / 2)
            )

            resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_POSITION, position.matrix)
            resources.model_sprite.render()

        for j in range(2):
            _, texture = resources.sprite_sets[self._sprite_set].get_texture(6 + j, 0)

            texture.bind(0)

            position = ProjectionMatrix().scale(
                (bounds.bounds[0] + self._outline_thickness, self._outline_thickness)
            ).translate(
                (bounds.position[0] + bounds.bounds[0] / 2, bounds.position[1] + j * bounds.bounds[1])
            )

            resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_POSITION, position.matrix)
            resources.model_sprite.render()

        _, texture = resources.sprite_sets[self._sprite_set].get_texture(8, 0)

        texture.bind(0)

        position = ProjectionMatrix().scale(
            bounds.bounds
        ).translate(
            bounds.position + bounds.bounds / 2
        )

        resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_POSITION, position.matrix)
        resources.model_sprite.render()


class GUIComponent:
    """
    The top level GUI object.

    Any component should extend from this class.

    Attributes
    ----------
    bounds: AxisAlignedBoundingBox
        The bounding box of the component.
    max_size: numpy.ndarray
        The maximum size vector of the component.
    border: GUIBorder
        The border of the component.
    visible: bool
        Disables the rendering of the component is set to False.
    allow_focus: bool
        Allows the component to get the focus is set to True.
    focused: bool
        If set to True, the component has the focus in the GUI.

    Methods
    -------
    render(graphics, projection)
        Renders the component.
    fire_events(tick, event_queue)
        Fires the GUI events.
    give_focus(pointer, projection)
        Tries to give the focus to the component.
    reset_focus()
        Resets the focus of the component.
    """

    def __init__(self, bounds: AxisAlignedBoundingBox, max_size: [tuple, numpy.ndarray] = None,
                 border: GUIBorder = None, visible: bool = True, allow_focus: bool = False):
        """
        Initializes the GUIComponent.

        Parameters
        ----------
        bounds: AxisAlignedBoundingBox
            The bounding box of the component.
        max_size: [tuple, numpy.ndarray], optional
            The maximum size vector of the component.
        border: GUIBorder, optional
            The border of the component.
        visible: bool, optional
            Disables the rendering of the component is set to False.
        allow_focus: bool, optional
            Allows the component to get the focus is set to True.
        """

        self.bounds = bounds

        if max_size is None:
            max_size = bounds.bounds

        self.max_size = array_format(max_size, dtype=numpy.float32)

        self.border = border

        self.visible = visible
        self.allow_focus = allow_focus

        self.focused = False
        self._focused = False

    def render(self, resources: ResourceManager, projection: ProjectionMatrix) -> None:
        """
        Renders the component.

        Renders the component accordingly to the given projection matrix. This matrix is translated to the origin of
        the component, hence only a translation to the position of the bounding box is needed. This is more convenient
        for nested objects. By default, this function only renders the border, it should be overridden by the object
        extending from this class.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        projection: ProjectionMatrix
            The local projection matrix.
        """

        if self.visible:
            if self.border is not None:
                rendered_bounds = AxisAlignedBoundingBox(
                    self.bounds.position, numpy.min([self.bounds.bounds, self.max_size], axis=0)
                )

                self.border.render(rendered_bounds, resources, projection)

    def fire_events(self, tick: int, event_queue: EventQueue) -> None:
        """
        Fires the GUI events.

        Fires the events related to the GUI state. By default, this function only fires events related to the focus, it
        should be overridden by the objects extending from this class.

        Parameters
        ----------
        tick: int
            The current logic tick.
        event_queue:
            The main event queue used to handle events.
        """

        if not self.allow_focus:
            self.focused = False

        if self._focused != self.focused:
            if self.focused:
                event_queue.fire_event(GUIFocusedEvent(tick, self))
            else:
                event_queue.fire_event(GUIUnfocusedEvent(tick, self))

            self._focused = self.focused

    def give_focus(self, pointer: numpy.ndarray, projection: ProjectionMatrix) -> bool:
        """
        Tries to give the focus to the component.

        Tests if the pointers is in the bounding box of the component. If it is the case, the focus should be given to
        this component.

        Parameters
        ----------
        pointer: numpy.ndarray
            The position of the mouse pointer on the screen.
        projection: ProjectionMatrix
            The local projection matrix.

        Returns
        -------
        got_focus: bool
            True if the component got the focus and False otherwise.
        """

        if self.allow_focus:
            local_pointer = transform_gui(pointer, projection)

            bounds = AxisAlignedBoundingBox(
                self.bounds.position, numpy.min([self.bounds.bounds, self.max_size], axis=0)
            )

            self.focused = bounds.inner_point(local_pointer)

            return self.focused

        return False

    def reset_focus(self) -> None:
        """
        Resets the focus of the component.
        """

        self.focused = False


class GUILayout:
    """
    The top level layout object.

    A layout is used to display several components. It is attached to a container and specifies how and where the sub
    components are rendered.

    Attributes
    ----------
    bounds: numpy.ndarray
        The size vector of the container.
    children: list of GUIComponent
        The component of the layout

    Methods
    -------
    render(graphics, projection)
        Renders the components of the layout.
    give_focus(pointer, projection)
        Tries to give the focus to a component of the layout.
    reset_focus()
        Resets the focus of each child.
    update_children()
        Updates the maximum size of the children.
    add_child(component)
        Adds a new component to the layout.
    clear()
        Clears the layout from its components.
    """

    def __init__(self, bounds: [tuple, numpy.ndarray] = None):
        """
        Initializes the GUILayout.

        Parameters
        ----------
        bounds: [tuple, numpy.ndarray], optional
            The size vector of the container.
        """

        if bounds is None:
            bounds = (0, 0)

        self.bounds = array_format(bounds, dtype=numpy.float32)
        self.children = []

    def render(self, resources: ResourceManager, projection: ProjectionMatrix) -> None:
        """
        Renders the components of the layout.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        projection: ProjectionMatrix
            The local projection matrix.
        """

    def give_focus(self, pointer: numpy.ndarray, projection: ProjectionMatrix) -> bool:
        """
        Tries to give the focus to a component of the layout.

        Parameters
        ----------
        pointer: numpy.ndarray
            The position of the mouse pointer on the screen.
        projection: ProjectionMatrix
            The local projection matrix.

        Returns
        -------
        got_focus: bool
            True if a component got the focus and False otherwise.
        """

    def reset_focus(self) -> None:
        """
        Resets the focus of each child.
        """

        for component in self.children:
            component.focused = False

    def update_children(self) -> None:
        """
        Updates the maximum size of the children.
        """

        for component in self.children:
            component.max_size = self.bounds - component.bounds.position

    def add_child(self, component: GUIComponent) -> None:
        """
        Adds a new component to the layout.

        Parameters
        ----------
        component: GUIComponent
            The new component.
        """

        self.children.append(component)

    def clear(self) -> None:
        """
        Clears the layout from its components.
        """

        self.children = []


class GUIAbsoluteLayout(GUILayout):
    """
    The default GUI layout.

    The layout places the components freely, in absolute position.

    Attributes
    ----------
    bounds: numpy.ndarray
        The size vector of the container.
    children: list of GUIComponent
        The component of the layout

    Methods
    -------
    render(graphics, projection)
        Renders the components of the layout.
    give_focus(pointer, projection)
        Tries to give the focus to a component of the layout.
    reset_focus()
        Resets the focus of each child.
    update_children()
        Updates the maximum size of the children.
    add_child(component)
        Adds a new component to the layout.
    clear()
        Clears the layout from its components.
    """

    def render(self, resources: ResourceManager, projection: ProjectionMatrix) -> None:
        """
        Renders the components of the layout.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        projection: ProjectionMatrix
            The local projection matrix.
        """

        for component in self.children:
            component.render(resources, projection)

    def give_focus(self, pointer: numpy.ndarray, projection: ProjectionMatrix) -> bool:
        """
        Tries to give the focus to a component of the layout.

        Parameters
        ----------
        pointer: numpy.ndarray
            The position of the mouse pointer on the screen.
        projection: ProjectionMatrix
            The local projection matrix.

        Returns
        -------
        got_focus: bool
            True if a component got the focus and False otherwise.
        """

        for component in self.children:
            if component.give_focus(pointer, projection):
                return True

        return False


class GUIListLayout(GUILayout):
    """
    A layout to display components vertically.

    This layout places the components one after each other from top to bottom. The positions of the components represent
    the offsets by which the components are moved when rendered.

    Attributes
    ----------
    bounds: numpy.ndarray
        The size vector of the container.
    children: list of GUIComponent
        The component of the layout
    padding: float
        The padding between each elements.

    Methods
    -------
    render(graphics, projection)
        Renders the components of the layout.
    give_focus(pointer, projection)
        Tries to give the focus to a component of the layout.
    reset_focus()
        Resets the focus of each child.
    update_children()
        Updates the maximum size of the children.
    add_child(component)
        Adds a new component to the layout.
    clear()
        Clears the layout from its components.
    """

    def __init__(self, bounds: [tuple, numpy.ndarray] = None, padding: float = 2.0):
        """
        Initializes the GUIListLayout.

        Parameters
        ----------
        bounds: [tuple, numpy.ndarray], optional
            The size vector of the container.
        padding: float, optional
            The padding between each elements.
        """

        GUILayout.__init__(self, bounds)

        self.padding = padding

    def render(self, resources: ResourceManager, projection: ProjectionMatrix) -> None:
        """
        Renders the components of the layout.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        projection: ProjectionMatrix
            The local projection matrix.
        """

        position = ProjectionMatrix().translate((self.padding, self.bounds[1]))

        for component in self.children:
            if component.visible:
                position.translate((0, - component.bounds.bounds[1] - self.padding))
                component.render(resources, ProjectionMatrix(position.matrix).dot(projection.matrix))

    def give_focus(self, pointer: numpy.ndarray, projection: ProjectionMatrix) -> bool:
        """
        Tries to give the focus to a component of the layout.

        Parameters
        ----------
        pointer: numpy.ndarray
            The position of the mouse pointer on the screen.
        projection: ProjectionMatrix
            The local projection matrix.

        Returns
        -------
        got_focus: bool
            True if a component got the focus and False otherwise.
        """

        position = ProjectionMatrix().translate((self.padding, self.bounds[1]))

        for component in self.children:
            if component.visible:
                position.translate((0, - component.bounds.bounds[1] - self.padding))

                if component.give_focus(pointer, ProjectionMatrix(position.matrix).dot(projection.matrix)):
                    return True

        return False

    def update_children(self) -> None:
        """
        Updates the maximum size of the children.
        """

        for component in self.children:
            component.max_size = (self.bounds - component.bounds.position) - 2 * self.padding


class GUIContainer(GUIComponent):
    """

    Attributes
    ----------
    bounds: AxisAlignedBoundingBox
        The bounding box of the component.
    max_size: numpy.ndarray
        The maximum size vector of the component.
    border: GUIBorder
        The border of the component.
    visible: bool
        Disables the rendering of the component is set to False.
    allow_focus: bool
        Allows the component to get the focus is set to True.
    focused: bool
        If set to True, the component has the focus in the GUI.

    Methods
    -------
    render(graphics, projection)
        Renders the component.
    fire_events(tick, event_queue)
        Fires the GUI events.
    give_focus(pointer, projection)
        Tries to give the focus to the component.
    reset_focus()
        Resets the focus of the component.
    get_focused()
        Returns the focused component of the container if there is any.
    add_child(component)
        Adds a new component to the container.
    clear()
        Clears the container from its components.
    """

    def __init__(self, bounds: AxisAlignedBoundingBox, layout: GUILayout, max_size: [tuple, numpy.ndarray] = None,
                 border: GUIBorder = None, visible: bool = True, allow_focus: bool = False):
        """
        Initializes the GUIContainer.

        Parameters
        ----------
        bounds: AxisAlignedBoundingBox
            The bounding box of the component.
        layout: GUILayout
            The layout used to place the sub components.
        max_size: [tuple, numpy.ndarray], optional
            The maximum size vector of the component.
        border: GUIBorder, optional
            The border of the component.
        visible: bool, optional
            Disables the rendering of the component is set to False.
        allow_focus: bool, optional
            Allows the component to get the focus is set to True.
        """

        GUIComponent.__init__(self, bounds, max_size, border, visible, allow_focus)

        self._layout = layout

        self._update_layout()

    def _update_layout(self) -> None:
        """
        Updates the size of the layout.
        """

        self._layout.bounds = numpy.min([self.bounds.bounds, self.max_size], axis=0)
        self._layout.update_children()

    def render(self, resources: ResourceManager, projection: ProjectionMatrix) -> None:
        """
        Renders the container.

        Renders the container accordingly to the given projection matrix. This matrix is translated to the origin of
        the component, hence only a translation to the position of the bounding box is needed. This is more convenient
        for nested objects.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        projection: ProjectionMatrix
            The local projection matrix.
        """

        self._update_layout()

        super().render(resources, projection)

        if self.visible:
            inner_projection = ProjectionMatrix().translate(self.bounds.position).dot(projection.matrix)

            self._layout.render(resources, inner_projection)

    def fire_events(self, tick: int, event_queue: EventQueue) -> None:
        """
        Fires the GUI events.

        Fires the events related to the GUI state. This also fires the events of the children.

        Parameters
        ----------
        tick: int
            The current logic tick.
        event_queue:
            The main event queue used to handle events.
        """

        super().fire_events(tick, event_queue)

        for child in self._layout.children:
            child.fire_events(tick, event_queue)

    def give_focus(self, pointer: numpy.ndarray, projection: ProjectionMatrix) -> bool:
        """
        Tries to give the focus to the component.

        Tests if the pointers is in the bounding box of the component. If it is the case, the focus should be given to
        this component.

        Parameters
        ----------
        pointer: numpy.ndarray
            The position of the mouse pointer on the screen.
        projection: ProjectionMatrix
            The local projection matrix.

        Returns
        -------
        got_focus: bool
            True if the component got the focus and False otherwise.
        """

        self.reset_focus()

        inner_projection = ProjectionMatrix().translate(self.bounds.position).dot(projection.matrix)

        if self._layout.give_focus(pointer, inner_projection):
            self.focused = False

            return True

        return super().give_focus(pointer, projection)

    def reset_focus(self) -> None:
        """
        Resets the focus of the component.
        """

        self._layout.reset_focus()

        super().reset_focus()

    def get_focused(self) -> GUIComponent:
        """
        Returns the focused component of the container if there is any.

        Returns
        -------
        component: GUIComponent
            The focused component if there is any.
        """

        if self.focused:
            return self

        for child in self._layout.children:
            if isinstance(child, GUIContainer):
                focus = child.get_focused()

                if focus is not None:
                    return focus
            elif child.focused:
                return child

    def add_child(self, component: GUIComponent) -> None:
        """
        Adds a new component to the container.

        Parameters
        ----------
        component: GUIComponent
            The new component.
        """

        self._layout.add_child(component)

    def clear(self) -> None:
        """
        Clears the container from its components.
        """

        self._layout.clear()


class GUILabel(GUIComponent):
    """
    A component used to display text.

    The label is used to display static text on the screen.

    Attributes
    ----------
    bounds: AxisAlignedBoundingBox
        The bounding box of the component.
    text: str
        The text of the label.
    font: GUIFont
        The font used to render the text.
    max_size: numpy.ndarray
        The maximum size vector of the component.
    border: GUIBorder
        The border of the component.
    visible: bool
        Disables the rendering of the component is set to False.
    allow_focus: bool
        Allows the component to get the focus is set to True.
    focused: bool
        If set to True, the component has the focus in the GUI.

    Methods
    -------
    render(graphics, projection)
        Renders the component.
    fire_events(tick, event_queue)
        Fires the GUI events.
    give_focus(pointer, projection)
        Tries to give the focus to the component.
    reset_focus()
        Resets the focus of the component.
    """

    def __init__(self, bounds: AxisAlignedBoundingBox, text: str, font: GUIFont,
                 max_size: [tuple, numpy.ndarray] = None, border: GUIBorder = None, visible: bool = True,
                 allow_focus: bool = False):
        """
        Initializes the GUILabel.

        Parameters
        ----------
        bounds: AxisAlignedBoundingBox
            The bounding box of the component.
        text: str
            The text of the label.
        font: GUIFont
            The font used to render the text.
        max_size: [tuple, numpy.ndarray], optional
            The maximum size vector of the component.
        border: GUIBorder, optional
            The border of the component.
        visible: bool, optional
            Disables the rendering of the component is set to False.
        allow_focus: bool, optional
            Allows the component to get the focus is set to True.
        """

        GUIComponent.__init__(self, bounds, max_size, border, visible, allow_focus)

        self.text = text

        self.font = font

    def render(self, resources: ResourceManager, projection: ProjectionMatrix) -> None:
        """
        Renders the label.

        Renders the label accordingly to the given projection matrix. Note that if the text is too long, it will be
        shortened to fit the bounding box.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        projection: ProjectionMatrix
            The local projection matrix.
        """

        super().render(resources, projection)

        if self.visible:
            resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_PROJECTION, projection.matrix)

            shown = int(min(self.bounds.bounds[0], self.max_size[0]) / self.font.width)
            displayed = self.text

            if shown < len(self.text):
                displayed = self.text[:shown - 3] + "..."

            if shown < 3:
                displayed = "." * shown

            for k in range(len(displayed)):
                char = displayed[k]

                _, texture = resources.sprite_sets[self.font.sprite_set].get_texture(
                    self.font.get_id_animation(char), 0
                )

                texture.bind(0)

                position = ProjectionMatrix().scale(
                    (self.font.width, self.font.height)
                ).translate((
                    self.bounds.position[0] + k * self.font.width + self.font.width / 2,
                    self.bounds.position[1] + self.font.height / 2
                ))

                resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_POSITION, position.matrix)
                resources.model_sprite.render()


class GUIImage(GUIComponent):
    """
    A component used to display sprites.

    The image is used to display static sprite on the screen.

    Attributes
    ----------
    bounds: AxisAlignedBoundingBox
        The bounding box of the component.
    sprite_set: str
        The sprite set containing the image to render.
    id_animation: int
        The index of the animation rendered.
    max_size: numpy.ndarray
        The maximum size vector of the component.
    border: GUIBorder
        The border of the component.
    visible: bool
        Disables the rendering of the component is set to False.
    allow_focus: bool
        Allows the component to get the focus is set to True.
    focused: bool
        If set to True, the component has the focus in the GUI.

    Methods
    -------
    render(graphics, projection)
        Renders the component.
    fire_events(tick, event_queue)
        Fires the GUI events.
    give_focus(pointer, projection)
        Tries to give the focus to the component.
    reset_focus()
        Resets the focus of the component.
    """

    def __init__(self, bounds: AxisAlignedBoundingBox, sprite_set: str, id_animation: int,
                 max_size: [tuple, numpy.ndarray] = None, border: GUIBorder = None, visible: bool = True,
                 allow_focus: bool = False):
        """
        Initializes the GUIImage.

        Parameters
        ----------
        bounds: AxisAlignedBoundingBox
            The bounding box of the component.
        sprite_set: str
            The sprite set containing the image to render.
        id_animation: int
            The index of the animation rendered.
        max_size: [tuple, numpy.ndarray], optional
            The maximum size vector of the component.
        border: GUIBorder, optional
            The border of the component.
        visible: bool, optional
            Disables the rendering of the component is set to False.
        allow_focus: bool, optional
            Allows the component to get the focus is set to True.
        """

        GUIComponent.__init__(self, bounds, max_size, border, visible, allow_focus)

        self.sprite_set = sprite_set
        self.id_animation = id_animation

    def render(self, resources: ResourceManager, projection: ProjectionMatrix) -> None:
        """
        Renders the image.

        Renders the image accordingly to the given projection matrix. Note that the sprite will be stretched in every
        direction to match the size of the bounding box.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        projection: ProjectionMatrix
            The local projection matrix.
        """

        super().render(resources, projection)

        if self.visible:
            _, texture = resources.sprite_sets[self.sprite_set].get_texture(self.id_animation, 0)

            texture.bind(0)

            bounds = numpy.min([self.bounds.bounds, self.max_size], axis=0)

            position = ProjectionMatrix().scale(
                bounds
            ).translate(
                self.bounds.position + bounds / 2
            )

            resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_PROJECTION, projection.matrix)
            resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_POSITION, position.matrix)
            resources.model_sprite.render()


class GUITextField(GUIComponent):
    """
    A component used to input text.

    The text field is an object in which the user can input text.

    Attributes
    ----------
    bounds: AxisAlignedBoundingBox
        The bounding box of the component.
    text: str
        The text of the label.
    font: GUIFont
        The font used to render the text.
    max_size: numpy.ndarray
        The maximum size vector of the component.
    border: GUIBorder
        The border of the component.
    visible: bool
        Disables the rendering of the component is set to False.
    allow_focus: bool
        Allows the component to get the focus is set to True.
    focused: bool
        If set to True, the component has the focus in the GUI.

    Methods
    -------
    render(graphics, projection)
        Renders the component.
    fire_events(tick, event_queue)
        Fires the GUI events.
    give_focus(pointer, projection)
        Tries to give the focus to the component.
    reset_focus()
        Resets the focus of the component.
    input_key(key_code)
        Appends a new character to the text.
    """

    def __init__(self, bounds: AxisAlignedBoundingBox, text: str, font: GUIFont,
                 max_size: [tuple, numpy.ndarray] = None, border: GUIBorder = None, visible: bool = True,
                 allow_focus: bool = True):
        """
        Initializes the GUITextField.

        Parameters
        ----------
        bounds: AxisAlignedBoundingBox
            The bounding box of the component.
        text: str
            The text of the label.
        font: GUIFont
            The font used to render the text.
        max_size: [tuple, numpy.ndarray], optional
            The maximum size vector of the component.
        border: GUIBorder, optional
            The border of the component.
        visible: bool, optional
            Disables the rendering of the component is set to False.
        allow_focus: bool, optional
            Allows the component to get the focus is set to True.
        """

        GUIComponent.__init__(self, bounds, max_size, border, visible, allow_focus)

        self.text = text
        self.font = font

    def render(self, resources: ResourceManager, projection: ProjectionMatrix) -> None:
        """
        Renders the label.

        Renders the label accordingly to the given projection matrix. Note that if the text is too long, only the last
        characters will be shown.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        projection: ProjectionMatrix
            The local projection matrix.
        """

        super().render(resources, projection)

        if self.visible:
            resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_PROJECTION, projection.matrix)

            shown = int(min(self.bounds.bounds[0], self.max_size[0]) / self.font.width)

            if shown == 0:
                return

            display = self.text[- shown:]

            for k in range(len(display)):
                char = display[k]

                _, texture = resources.sprite_sets[self.font.sprite_set].get_texture(
                    self.font.get_id_animation(char), 0
                )

                texture.bind(0)

                position = ProjectionMatrix().scale(
                    (self.font.width, self.font.height)
                ).translate((
                    self.bounds.position[0] + k * self.font.width + self.font.width / 2,
                    self.bounds.position[1] + self.font.height / 2
                ))

                resources.shader_sprite.set_uniform(ShaderProgram.UNIFORM_POSITION, position.matrix)
                resources.model_sprite.render()

    def input_key(self, key_code: int) -> None:
        """
        Appends a new character to the text.

        Parameters
        ----------
        key_code: int
            The code of the key input.
        """

        if key_code == Key.KEY_BACKSPACE and len(self.text) > 0:
            self.text = self.text[:-1]
        else:
            char = Key.get_corresponding_char(key_code)

            if char is not None:
                self.text += char


class GUIEvent(CancelableEvent):
    """
    A generic type of event used for GUI updates.

    This type of event is used for any event related to a GUI update. Theses event extend from CancelableEvent, hence
    they can be canceled while being processed by the event queue. This can be useful if more than one GUI components
    are updated at once, by giving priority to one over the others.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    component: GUIComponent
            The component which got updated.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """

    def __init__(self, tick: int, component: GUIComponent):
        """
        Initializes the GUIEvent.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        component: GUIComponent
            The component which got updated.
        """

        CancelableEvent.__init__(self, tick)

        self.component = component


class GUIFocusedEvent(GUIEvent):
    """
    The type of GUIEvent fired when a component gets focused.

    This kind of event is fired whenever a component gets focused.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    component: GUIComponent
            The component which got updated.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """


class GUIUnfocusedEvent(GUIEvent):
    """
    The type of GUIEvent fired when a component gets unfocused.

    This kind of event is fired whenever a component gets unfocused.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    component: GUIComponent
            The component which got updated.

    Methods
    -------
    cancel()
        Cancels the event.
    is_canceled()
        Returns whether or not the event got canceled.
    """


class GUIManager:
    """
    The main GUI holder.

    This is the entry point of the GUI. It contains the main container.

    Methods
    -------
    render()
        Renders the GUI.
    fire_events(tick)
        Fires the GUI events.
    add_child(component)
        Adds a new component to the GUI.
    give_focus(pointer)
        Tries to give the focus to a component of the GUI.
    reset_focus()
        Resets the focus of the GUI.
    get_focused()
        Returns the focused component of the GUI if there is any.
    clear()
        Clears the GUI from its components.
    default_handler_gui_click(event)
        The default mouse click handler.
    """

    def __init__(self, resources: ResourceManager, event_queue: EventQueue, viewport: [tuple, numpy.ndarray]):
        """
        Initializes the GUIManager.

        Parameters
        ----------
        resources: ResourceManager
            The main resource manager used for the rendering.
        event_queue: EventQueue
            The main event queue used to handle events.
        viewport: [tuple, numpy.ndarray]
            The size vector representing the viewport of the camera.
        """

        self._resources = resources
        self._event_queue = event_queue

        self._viewport = array_format(viewport, dtype=numpy.float32)

        self._projection = ProjectionMatrix().orthographic(
            self._viewport[0], - self._viewport[0], - self._viewport[1], self._viewport[1]
        ).translate(
            (-1, -1)
        )

        self._container = GUIContainer(AxisAlignedBoundingBox((0, 0), self._viewport * 2), GUIAbsoluteLayout())

    def render(self) -> None:
        """
        Renders the GUI.
        """

        self._container.render(self._resources, self._projection)

    def fire_events(self, tick: int) -> None:
        """
        Fires the GUI events.

        Parameters
        ----------
        tick: int
            The current logic tick.
        """

        self._container.fire_events(tick, self._event_queue)

    def add_child(self, component: GUIComponent) -> None:
        """
        Adds a new component to the GUI.

        Parameters
        ----------
        component: GUIComponent
            The new component.
        """

        self._container.add_child(component)

    def give_focus(self, pointer: numpy.ndarray) -> bool:
        """
        Tries to give the focus to a component of the GUI.

        Tests if the pointers is in the bounding box of a component. If it is the case, the focus should be given to
        this component.

        Parameters
        ----------
        pointer: numpy.ndarray
            The position of the mouse pointer on the screen.

        Returns
        -------
        got_focus: bool
            True if a component got the focus and False otherwise.
        """

        return self._container.give_focus(pointer, self._projection)

    def reset_focus(self) -> None:
        """
        Resets the focus of the GUI.
        """

        self._container.reset_focus()

    def get_focused(self) -> GUIComponent:
        """
        Returns the focused component of the GUI if there is any.

        Returns
        -------
        component: GUIComponent
            The focused component if there is any.
        """

        return self._container.get_focused()

    def clear(self) -> None:
        """
        Clears the GUI from its components.
        """

        self._container.clear()

    def default_handler_gui_click(self, event: MouseButtonClickedEvent) -> None:
        """
        The default mouse click handler.

        This handler is a default handler. It simply gives the focus of the GUI component clicked.

        Parameters
        ----------
        event: MouseButtonClickedEvent
            The mouse button event fired.
        """

        if event.button == MouseButton.MOUSE_BUTTON_LEFT:
            self.give_focus(event.position)
