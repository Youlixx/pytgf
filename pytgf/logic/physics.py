"""
Contains every classes related to the logic engine.
"""

from pytgf.logic.event import Event, EventQueue

from multiprocessing.pool import ThreadPool
from itertools import repeat
from time import sleep

import numpy
import datetime


def array_format(vector: [tuple, numpy.ndarray], dtype: type = numpy.int32) -> numpy.ndarray:
    """
    Converts the vector from any type to a numpy array.

    This function should be called each time an array is passed as an argument. It prevent array from being cast to
    other data type. It also converts tuples into numpy arrays.

    Parameters
    ----------
    vector: [tuple, numpy.ndarray]
        The vector to reformat. Either a tuple or a numpy array.
    dtype: type, optional
        The type of the output array.

    Returns
    -------
    array: numpy.ndarray
        The formatted vector in the specified data type.
    """

    if isinstance(vector, numpy.ndarray):
        return vector.astype(dtype)
    else:
        return numpy.array(vector, dtype=dtype)


class AxisAlignedBoundingBox:
    """
    The standard way of representing bounding boxes.

    An AxisAlignedBoundingBox (shortened AABB) represents a simple rectangle aligned with the x and y axis. The
    collision detection algorithm and the rendering revolve around these. The rectangle is stored withe the format
    (x, y, w, h) where (x, y) is the position of the bottom-left corner and (w, h) is its size.

    Attributes
    ----------
    position: numpy.ndarray
        The position of the bottom-left corner of the rectangle.
    bounds: numpy.ndarray
        The width and the height of the rectangle.

    Methods
    -------
    inner_point(point)
        Returns the inner point test.
    intersects(other)
        Returns whether or not two AABBs are intersecting.
    expand(displacement)
        Creates an expanded version of the AABB.
    copy()
        Creates a copy of the AABB.
    as_type(dtype)
        Creates a copy of the AABB and casts it to the specified type.
    """

    def __init__(self, position: [tuple, numpy.ndarray], bounds: [tuple, numpy.ndarray], dtype: type = numpy.int32):
        """
        Initializes the AxisAlignedBoundingBox.

        Parameters
        ----------
        position: [tuple, numpy.ndarray]
            The position of the bottom-left corner of the rectangle.
        bounds: [tuple, numpy.ndarray]
            The width and the height of the rectangle.
        dtype; type, optional
            The data format in which the variables are stored.
        """

        self._dtype = dtype

        self._position = array_format(position, dtype=dtype)
        self._bounds = array_format(bounds, dtype=dtype)

    @property
    def position(self) -> numpy.ndarray:
        """
        The position property representing the position of the bottom-left corner of the rectangle.
        """

        return self._position

    @position.setter
    def position(self, position: [tuple, numpy.ndarray]) -> None:
        """
        Setter function for the position vector.

        Parameters
        ----------
        position: [tuple, numpy.ndarray]
            The position of the bottom-left corner of the rectangle.
        """

        self._position = array_format(position, dtype=self._dtype)

    @property
    def bounds(self) -> numpy.ndarray:
        """
        The bounds property representing the width and the height of the rectangle.
        """

        return self._bounds

    @bounds.setter
    def bounds(self, bounds: [tuple, numpy.ndarray]) -> None:
        """
        Setter function for the bounds vector.

        Parameters
        ----------
        bounds: [tuple, numpy.ndarray]
            The width and the height of the rectangle.
        """

        self._bounds = array_format(bounds, dtype=self._dtype)

    def __contains__(self, other: "AxisAlignedBoundingBox") -> bool:
        """
        Returns whether or not the other AABB is in the main one

        Tests if the specified AABB is in the main one. It returns True only if the other embedded in the main one.

        Parameters
        ----------
        other: AxisAlignedBoundingBox
            The other AABB with which the test is performed.

        Returns
        -------
        result: bool
            True if the tested AABB is in the main one and False otherwise.
        """

        return numpy.logical_and(
            * (self.position <= other.position) * (self.position + self.bounds >= other.position + self.bounds)
        )

    def inner_point(self, point: numpy.ndarray) -> bool:
        """
        Returns the inner point test.

        Parameters
        ----------
        point: numpy.ndarray
            The point with which the test is performed.

        Returns
        -------
        result: bool
            True if the tested point is in the AABB and False otherwise.
        """

        relative_position = point.astype(self._dtype) - self.position

        return numpy.logical_and(*(0 <= relative_position) * (relative_position <= self.bounds))

    def intersects(self, other: "AxisAlignedBoundingBox") -> bool:
        """
        Returns whether or not two AABBs are intersecting.

        Parameters
        ----------
        other: AxisAlignedBoundingBox
            The other AABB with which the test is performed.

        Returns
        -------
        result: bool
            True if the two AABBs are intersecting and False otherwise.
        """

        return not (
            numpy.logical_or(*(self.position >= other.position + other.bounds) +
                              (self.position + self.bounds <= other.position))
        )

    def expand(self, displacement: [tuple, numpy.ndarray]) -> "AxisAlignedBoundingBox":
        """
        Creates an expanded version of the AABB.

        The AABB can be expanded in every direction. The size of the AABB is increased by the displacement vector; if
        a component is negative along an axis, the position is moved by the displacement.

        Parameters
        ----------
        displacement: [tuple, numpy.ndarray]
            The amount by which the AABB is expanded in each direction.

        Returns
        -------
        aabb: AxisAlignedBoundingBox
            The expanded AABB.
        """

        displacement = array_format(displacement, dtype=self._dtype)

        return AxisAlignedBoundingBox(
            self.position + numpy.clip(displacement, a_min=None, a_max=0),
            self.bounds + numpy.abs(displacement),
            self._dtype
        )

    def copy(self) -> "AxisAlignedBoundingBox":
        """
        Creates a copy of the AABB.

        Returns
        -------
        aabb: AxisAlignedBoundingBox
            The copied AABB.
        """

        return AxisAlignedBoundingBox(self.position, self.bounds, dtype=self._dtype)

    def as_type(self, dtype: type) -> "AxisAlignedBoundingBox":
        """
        Creates a copy of the AABB and casts it to the specified type.

        Parameters
        ----------
        dtype: type
            The data format in which the variables of the copy are stored.

        Returns
        -------
        aabb: AxisAlignedBoundingBox
            The copied AABB in the new format.
        """

        return AxisAlignedBoundingBox(self.position, self.bounds, dtype=dtype)


class WorldObject:
    """
    The top level world object type

    Any object that can be spawned in the world must extends from WorldObject. However, bare WorldObject should not be
    used. For object subject to collision, consider using the Entity class and for visual effect use Particle.

    Attributes
    ----------
    bounding_box: AxisAlignedBoundingBox
        The bounding box of the object.
    position: numpy.ndarray
        The position of the bottom-left corner of the rectangle.
    bounds: numpy.ndarray
        The width and the height of the rectangle.
    """

    def __init__(self, bounding_box: AxisAlignedBoundingBox):
        """
        Initializes the WorldObject.

        Parameters
        ----------
        bounding_box: AxisAlignedBoundingBox
            The bounding box of the object.
        """

        self.bounding_box = bounding_box

        self.should_be_destroyed = False

    @property
    def position(self) -> numpy.ndarray:
        """
        The position property representing the position of the bottom-left corner of the rectangle.
        """

        return self.bounding_box.position

    @position.setter
    def position(self, position: [tuple, numpy.ndarray]) -> None:
        """
        Setter function for the position vector.

        Parameters
        ----------
        position: [tuple, numpy.ndarray]
            The position of the bottom-left corner of the rectangle.
        """

        self.bounding_box.position = position

    @property
    def bounds(self) -> numpy.ndarray:
        """
        The bounds property representing the width and the height of the rectangle.
        """

        return self.bounding_box.bounds

    @bounds.setter
    def bounds(self, bounds: [tuple, numpy.ndarray]) -> None:
        """
        Setter function for the bounds vector.

        Parameters
        ----------
        bounds: [tuple, numpy.ndarray]
            The width and the height of the rectangle.
        """

        self.bounding_box.bounds = bounds


class PhysicsObject(WorldObject):
    """
    A generic type of world object used for collision detection.

    An object that collides with tiles either and / or other objects should extends from this class. Note that theses
    objects are not supposed to move, extend from Entity instead.

    Attributes
    ----------
    bounding_box: AxisAlignedBoundingBox
        The bounding box of the object.
    position: numpy.ndarray
        The position of the bottom-left corner of the rectangle.
    bounds: numpy.ndarray
        The width and the height of the rectangle.
    collides_with_tiles: bool
        Enables the collision detection with tiles.
    colliders: list of types
        The list of types of object with which the object collides.

    Methods
    -------
    add_collider(collider)
        Adds a new collider.
    remove_collider(collider)
        Removes a collider.
    clear_colliders()
        Removes every existing collider.
    """

    def __init__(self, bounding_box: AxisAlignedBoundingBox, collides_with_tiles: bool = True,
                 colliders: [type, list] = None):
        """
        Initializes the PhysicsObject.

        Parameters
        ----------
        bounding_box: AxisAlignedBoundingBox
            The bounding box of the object.
        collides_with_tiles: bool, optional
            Enables the collision detection with tiles.
        colliders: [type, list], optional
            The list of types of object with which the object collides.
        """

        WorldObject.__init__(self, bounding_box)

        self.collides_with_tiles = collides_with_tiles

        self.colliders = colliders if colliders is not None else []

    @property
    def colliders(self) -> list:
        """
        The colliders property containing the list of types of object with which the object collides.
        """

        return self._colliders

    @colliders.setter
    def colliders(self, colliders: [type, list]) -> None:
        """
        Setter function for the colliders list.

        Parameters
        ----------
        colliders: [type, list]
             The list of types of object with which the object collides.

        Raises
        ------
        TypeError
            If the colliders list contains classes that are not extending from PhysicsObject
        """

        if type(colliders) == type and issubclass(colliders, PhysicsObject):
            self._colliders = [colliders]

        elif type(colliders) == list:
            for collider in colliders:
                if type(collider) != type or not issubclass(collider, PhysicsObject):
                    raise TypeError("The collider {0} is not a subclass of PhysicsObjects.".format(str(collider)))

            self._colliders = colliders

        else:
            raise TypeError("The colliders must be a list of subclass of PhysicsObjects.")

    def add_collider(self, collider: type) -> None:
        """
        Adds a new collider.

        Adds a new type of object with which the object will collide. Note that the type must be a subclass of
        PhysicsObject.

        Parameters
        ----------
        collider: type
            The type of object with which the object will collide.

        Raises
        ------
        TypeError
            If the colliders list contains classes that are not extending from PhysicsObject
        """

        if type(collider) != type or not issubclass(collider, PhysicsObject):
            raise TypeError("The collider {0} is not a subclass of PhysicsObjects.".format(str(collider)))

        if collider not in self.colliders:
            self.colliders.append(collider)

    def remove_collider(self, collider: type) -> None:
        """
        Removes a collider.

        Removes a type of object with which the world object used to collide.

        Parameters
        ----------
        collider: type
            The type of object with which the object will not collide anymore.
        """

        if collider in self.colliders:
            self.colliders.remove(collider)

    def clear_colliders(self) -> None:
        """
        Removes every existing collider.
        """

        self.colliders.clear()


class Renderable(WorldObject):
    """
    A generic type of world object used for rendering.

    An object that is supposed to be rendered by drawing a texture should extends from this class. For object that need
    to have a physics (collision detection and movement), see Entity.

    Attributes
    ----------
    bounding_box: AxisAlignedBoundingBox
        The bounding box of the object.
    position: numpy.ndarray
        The position of the bottom-left corner of the rectangle.
    bounds: numpy.ndarray
        The width and the height of the rectangle.
    texture_bounds: AxisAlignedBoundingBox
        The bounding box of the displayed texture. Note that the position of the sprite is relative to the position of
        the world object (a position of (0, 0) means that the sprite will be drawn at the same position of the object).
    sprite_set: str
        The name of the sprite set used.
    id_animation: int
        The index of the animation played within the sprite set.
    id_animation_transition: int
        The index of the animation played as a transition within the sprite set.
    angle: float
        The angle between the sprite and the
    visible: bool
        Renders the object if set to True.
    flip_horizontally: bool
        Flips the texture horizontally if set to True.
    flip_vertically: bool
        Flips the texture vertically if set to True.

    Methods
    -------
    transition(id_animation_transition, id_animation=-1)
        Plays a transition animation once.
    cancel_transition(id_animation=-1)
        Cancels the transition animation immediately.
    is_transiting()
        Returns whether or not a transition animation is playing.
    """

    def __init__(self, bounding_box: AxisAlignedBoundingBox, texture_bounds: AxisAlignedBoundingBox, sprite_set: str,
                 id_animation: int, angle: float = 0, visible: bool = True, flip_horizontally: bool = False,
                 flip_vertically: bool = False):
        """
        Initializes the Renderable.

        Parameters
        ----------
        bounding_box: AxisAlignedBoundingBox
            The bounding box of the object.
        texture_bounds: AxisAlignedBoundingBox
            The bounding box of the displayed texture. Note that the position of the sprite is relative to the position
            of the world object.
        sprite_set: str
            The name of the sprite set used.
        id_animation: int
            The index of the animation played within the specified sprite set.
        angle: float, optional
            The angle between the sprite and the
        visible: bool, optional
            Renders the object if set to True.
        flip_horizontally: bool, optional
            Flips the texture horizontally if set to True.
        flip_vertically: bool, optional
            Flips the texture vertically if set to True.
        """

        WorldObject.__init__(self, bounding_box)

        self.texture_bounds = texture_bounds

        self.sprite_set = sprite_set
        self.id_animation = id_animation
        self.id_animation_transition = -1

        self.animation_pointer = 0

        self.angle = angle
        self.visible = visible

        self.flip_horizontally = flip_horizontally
        self.flip_vertically = flip_vertically

    @property
    def animation_pointer(self) -> int:
        """
        The animation pointer property containing the current frame index of the played animation.
        """

        return self._animation_pointer

    @animation_pointer.setter
    def animation_pointer(self, animation_pointer: int) -> None:
        """
        Setter function for the animation pointer.

        Parameters
        ----------
        animation_pointer: int
             The new index of the frame to draw.
        """

        if self.id_animation_transition != -1 and animation_pointer == 0:
            self.id_animation_transition = -1

        self._animation_pointer = animation_pointer

    def transition(self, id_animation_transition: int, id_animation: int = -1) -> None:
        """
        Plays a transition animation once.

        Plays a transition animation which will go through a the full animation once before switching back to the main
        animation.

        Parameters
        ----------
        id_animation_transition: int
            The index of the transition animation played once.
        id_animation: int, optional
            The index of the animation played after the transition.
        """

        if id_animation != -1:
            self.id_animation = id_animation

        self.animation_pointer = 0
        self.id_animation_transition = id_animation_transition

    def cancel_transition(self, id_animation: int = -1) -> None:
        """
        Cancels the transition animation immediately.

        Cancels the transition animation before it ends. If a new animation is not specified, the transition cancels
        into the animation stated at the start of the transition.

        Parameters
        ----------
        id_animation: int, optional
            The index of the new animation played.
        """

        if id_animation != -1:
            self.id_animation = id_animation

        self.id_animation_transition = -1
        self.animation_pointer = 0

    def is_transiting(self) -> bool:
        """
        Returns whether or not a transition animation is playing.

        Returns
        -------
        transiting: bool
            True is a transition animation is currently played and False otherwise.
        """

        return self.id_animation_transition != -1


class Particle(Renderable):
    """
    This type of world object is used to render particles.

    A particle is an animation played once. When the animation cycle is completed, the object destroys itself.

    Attributes
    ----------
    bounding_box: AxisAlignedBoundingBox
        The bounding box of the object.
    position: numpy.ndarray
        The position of the bottom-left corner of the rectangle.
    bounds: numpy.ndarray
        The width and the height of the rectangle.
    texture_bounds: AxisAlignedBoundingBox
        The bounding box of the displayed texture. Note that the position of the sprite is relative to the position of
        the world object (a position of (0, 0) means that the sprite will be drawn at the same position of the object).
    sprite_set: str
        The name of the sprite set used.
    id_animation: int
        The index of the animation played within the specified sprite set.
    id_animation_transition: int
        The index of the animation played as a transition within the sprite set.
    angle: float
        The angle between the sprite and the
    visible: bool
        Renders the object if set to True.
    flip_horizontally: bool
        Flips the texture horizontally if set to True.
    flip_vertically: bool
        Flips the texture vertically if set to True.

    Methods
    -------
    transition(id_animation_transition, id_animation=-1)
        Plays a transition animation once.
    cancel_transition(id_animation=-1)
        Cancels the transition animation immediately.
    is_transiting()
        Returns whether or not a transition animation is playing.
    """

    def __init__(self, bounding_box: AxisAlignedBoundingBox, texture_bounds: AxisAlignedBoundingBox, sprite_set: str,
                 id_animation: int, angle: float = 0, visible: bool = True, flip_horizontally: bool = False,
                 flip_vertically: bool = False):
        """
        Initializes the Particle.

        Parameters
        ----------
        bounding_box: AxisAlignedBoundingBox
            The bounding box of the object.
        texture_bounds: AxisAlignedBoundingBox
            The bounding box of the displayed texture. Note that the position of the sprite is relative to the position
            of the world object.
        sprite_set: str
            The name of the sprite set used.
        id_animation: int
            The index of the animation played within the specified sprite set.
        angle: float, optional
            The angle between the sprite and the
        visible: bool, optional
            Renders the object if set to True.
        flip_horizontally: bool, optional
            Flips the texture horizontally if set to True.
        flip_vertically: bool, optional
            Flips the texture vertically if set to True.
        """

        self._spawned = False

        Renderable.__init__(
            self, bounding_box, texture_bounds, sprite_set, id_animation, angle=angle, visible=visible,
            flip_horizontally=flip_horizontally, flip_vertically=flip_vertically
        )

    @property
    def animation_pointer(self) -> int:
        """
        The animation pointer property containing the current frame index of the played animation.
        """

        return self._animation_pointer

    @animation_pointer.setter
    def animation_pointer(self, animation_pointer: int) -> None:
        """
        Setter function for the animation pointer.

        Parameters
        ----------
        animation_pointer: int
             The new index of the frame to draw.
        """

        if self.id_animation_transition != -1 and animation_pointer == 0:
            self.id_animation_transition = -1

        if self._spawned and animation_pointer == 0:
            self.visible = False
            self.should_be_destroyed = True

        self._animation_pointer = animation_pointer
        self._spawned = True


class Entity(PhysicsObject, Renderable):
    """
    This type of world object for renderable objects with collision detection.

    This is the main world object type used. An entity can be rendered, it can moves and it can collides with tiles and
    other entities.

    Attributes
    ----------
    bounding_box: AxisAlignedBoundingBox
        The bounding box of the object.
    position: numpy.ndarray
        The position of the bottom-left corner of the rectangle.
    bounds: numpy.ndarray
        The width and the height of the rectangle.
    speed: numpy.ndarray
        The speed vector of the entity expressed in unit per tick.
    texture_bounds: AxisAlignedBoundingBox
        The bounding box of the displayed texture. Note that the position of the sprite is relative to the position of
        the world object (a position of (0, 0) means that the sprite will be drawn at the same position of the object).
    sprite_set: str
        The name of the sprite set used.
    id_animation: int
        The index of the animation played within the specified sprite set.
    id_animation_transition: int
        The index of the animation played as a transition within the sprite set.
    angle: float
        The angle between the sprite and the
    visible: bool
        Renders the object if set to True.
    flip_horizontally: bool
        Flips the texture horizontally if set to True.
    flip_vertically: bool
        Flips the texture vertically if set to True.
    collides_with_tiles: bool
        Enables the collision detection with tiles.
    colliders: list of types
        The list of types of object with which the object collides.

    Methods
    -------
    transition(id_animation_transition, id_animation=-1)
        Plays a transition animation once.
    cancel_transition(id_animation=-1)
        Cancels the transition animation immediately.
    is_transiting()
        Returns whether or not a transition animation is playing.
    add_collider(collider)
        Adds a new collider.
    remove_collider(collider)
        Removes a collider.
    clear_colliders()
        Removes every existing collider.
    """

    def __init__(self, bounding_box: AxisAlignedBoundingBox, speed: [tuple, numpy.ndarray],
                 texture_bounds: AxisAlignedBoundingBox, sprite_set: str, id_animation: int, angle: float = 0,
                 visible: bool = True, flip_horizontally: bool = False, flip_vertically: bool = False,
                 collides_with_tiles: bool = True, colliders: [type, list] = None):
        """
        Initializes the Entity.

        Parameters
        ----------
        bounding_box: AxisAlignedBoundingBox
            The bounding box of the object.
        speed: [tuple, numpy.ndarray]
            The speed vector of the entity expressed in unit per tick.
        texture_bounds: AxisAlignedBoundingBox
            The bounding box of the displayed texture. Note that the position of the sprite is relative to the position
            of the world object.
        sprite_set: str
            The name of the sprite set used.
        id_animation: int
            The index of the animation played within the specified sprite set.
        angle: float, optional
            The angle between the sprite and the
        visible: bool, optional
            Renders the object if set to True.
        flip_horizontally: bool, optional
            Flips the texture horizontally if set to True.
        flip_vertically: bool, optional
            Flips the texture vertically if set to True.
        collides_with_tiles: bool, optional
            Enables the collision detection with tiles.
        colliders: [type, list], optional
            The list of types of object with which the object collides.
        """

        PhysicsObject.__init__(self, bounding_box, collides_with_tiles=collides_with_tiles, colliders=colliders)

        Renderable.__init__(
            self, bounding_box, texture_bounds, sprite_set, id_animation, angle=angle, visible=visible,
            flip_horizontally=flip_horizontally, flip_vertically=flip_vertically
        )

        self.speed = speed

    @property
    def speed(self) -> numpy.ndarray:
        """
        The speed property representing the speed vector of the entity expressed in unit per tick.
        """

        return self._speed

    @speed.setter
    def speed(self, speed: [tuple, numpy.ndarray]) -> None:
        """
        Setter function for the speed vector.

        Parameters
        ----------
        speed: [tuple, numpy.ndarray]
             The speed vector of the entity expressed in unit per tick.
        """

        self._speed = array_format(speed)


class CollisionMap:
    """
    Simple way of defining the behavior of a tile.

    Collisions are only considered on the edges of the tiles. The collision map allows to define how the entities will
    interact with each of the edge (either collide or go through).

    Attributes
    ----------
    collide_north: bool
        Enables the collision detection with the top edge of the tile if set to True.
    collide_east: bool
        Enables the collision detection with the right edge of the tile if set to True.
    collide_south: bool
        Enables the collision detection with the bottom edge of the tile if set to True.
    collide_west: bool
        Enables the collision detection with the left edge of the tile if set to True.
    """

    def __init__(self, collide_north: bool, collide_east: bool, collide_south: bool, collide_west: bool):
        """
        Initializes the CollisionMap.

        Parameters
        ----------
        collide_north: bool
            Enables the collision detection with the top edge of the tile if set to True.
        collide_east: bool
            Enables the collision detection with the right edge of the tile if set to True.
        collide_south: bool
            Enables the collision detection with the bottom edge of the tile if set to True.
        collide_west: bool
            Enables the collision detection with the left edge of the tile if set to True.
        """

        self.collide_north = collide_north
        self.collide_east = collide_east
        self.collide_south = collide_south
        self.collide_west = collide_west


class TileManager:
    """
    The tile physics manager.

    A tile manager must be created to registers tiles. A tile is a couple of an texture and a collision map. This object
    should be fully initialized before creating a world. To add the rendering, see the ResourceManager class. Note that
    the scale used here expresses the size of a tile in term of game distance unit.

    Attributes
    ----------
    tile_size: int
        The size of the tiles expressed in distance units.

    Methods
    -------
    register_collision_map(collision_map)
        Registers a new collision map.
    register_tile(id_collision_map, id_texture)
        Registers a new tile.
    get_collision_map(id_tile)
        Returns the collision map associated to the tile.
    get_id_texture(id_tile)
        Returns the texture associated to the tile.
    """

    def __init__(self, tile_size: int):
        """
        Initializes the TileManager.

        Parameters
        ----------
        tile_size: int
            The size of the tiles expressed in distance units.
        """

        self.tile_size = tile_size

        self._collision_maps = []
        self._tiles = []

    def register_collision_map(self, collision_map: CollisionMap) -> None:
        """
        Registers a new collision map.

        Registers a new collision map and gives it the first index available.

        Parameters
        ----------
        collision_map: CollisionMap
            The collision map describing the collision behavior of a tile.
        """

        self._collision_maps.append(collision_map)

    def register_tile(self, id_collision_map: int, id_texture: int) -> None:
        """
        Registers a new tile.

        Registers a new tile with the specified texture and collision map indexes and gives the first index available.
        Note that the index 0 corresponds to the empty tile (no rendering nor collision).

        Parameters
        ----------
        id_collision_map: int
            The index of the collision map used for the collision behavior.
        id_texture: int
            The index of the texture used (see ResourcesManagers)
        """

        self._tiles.append((id_collision_map, id_texture))

    def get_collision_map(self, id_tile: int) -> CollisionMap:
        """
        Returns the collision map associated to the tile.

        Returns the collision map associated to the tile registered at the specified index. Note that the index 0 is
        forbidden since it corresponds to the empty tile, nothing should collide with it.

        Parameters
        ----------
        id_tile: int
            The index of the tile.

        Returns
        -------
        collision_map: CollisionMap
            The collision map associated to the tile.
        """

        if not 0 < id_tile <= len(self._tiles):
            raise ValueError("The tile with ID " + str(id_tile) + " is not registered.")

        return self._collision_maps[self._tiles[id_tile - 1][0]]

    def get_id_texture(self, id_tile: int) -> int:
        """
        Returns the texture associated to the tile.

        Returns the index of the texture associated to the tile registered at the specified index. Note that the index 0
        is forbidden since it corresponds to the empty tile, it should not be rendered.

        Parameters
        ----------
        id_tile: int
            The index of the tile.

        Raises
        ------
        ValueError
            If the tile index is invalid.

        Returns
        -------
        id_texture: int
            The index of the texture to render.
        """

        if not 0 < id_tile <= len(self._tiles):
            raise ValueError("The tile with ID " + str(id_tile) + " is not registered.")

        return self._tiles[id_tile - 1][1]


class Direction:
    """
    An enumeration containing the different directions.

    The collisions directions are represented using theses direction codes.

    Methods
    -------
    next_right(direction)
        Gives the direction obtained by a 90 degrees rotation clockwise.
    next_left(direction)
        Gives the direction obtained by a 90 degrees rotation counterclockwise.
    opposite(direction)
        Gives the direction obtained by a 180 degrees rotation.
    """

    DIRECTION_NONE = -1
    DIRECTION_NORTH = 0
    DIRECTION_EAST = 1
    DIRECTION_SOUTH = 2
    DIRECTION_WEST = 3

    @staticmethod
    def next_right(direction: int) -> int:
        """
        Gives the direction obtained by a 90 degrees rotation clockwise.

        Parameters
        ----------
        direction
            The initial direction code.

        Returns
        -------
        direction: int
            The direction obtained after the rotation.
        """

        if direction == Direction.DIRECTION_NONE:
            return Direction.DIRECTION_NONE

        return (direction + 1) % 4

    @staticmethod
    def next_left(direction: int) -> int:
        """
        Gives the direction obtained by a 90 degrees rotation counterclockwise.

        Parameters
        ----------
        direction
            The initial direction code.

        Returns
        -------
        direction: int
            The direction obtained after the rotation.
        """

        if direction == Direction.DIRECTION_NONE:
            return Direction.DIRECTION_NONE

        return (direction + 3) % 4

    @staticmethod
    def opposite(direction: int) -> int:
        """
        Gives the direction obtained by a 180 degrees rotation.

        Parameters
        ----------
        direction
            The initial direction code.

        Returns
        -------
        direction: int
            The direction obtained after the rotation.
        """

        if direction == Direction.DIRECTION_NONE:
            return Direction.DIRECTION_NONE

        return (direction + 2) % 4


class CollisionEvent(Event):
    """
    A generic type of event used for collision.

    This type of event is used for any event related to a collision between an entity with either a tile or another
    entity. Theses events are not cancelable.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    entity: Entity
        The entity which collided.
    direction: int
        The direction of the collision relative to the entity.
    """
    
    def __init__(self, tick: int, entity: Entity, direction: int):
        """
        Initializes the CollisionEvent.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        entity: Entity
            The entity which collided.
        direction: int
            The direction of the collision relative to the entity.
        """
        
        Event.__init__(self, tick)
        
        self.entity = entity
        self.direction = direction


class CollisionWithTileEvent(CollisionEvent):
    """
    The type of CollisionEvent used for collisions between an entity and a tile.

    This kind of event is fired whenever an entity collides with a tile. Note that after being handled, the speed of the
    entity who collided must have changed to avoid colliding again with the tile (or else you are likely to be stuck in
    an infinite loop).

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    entity: Entity
        The entity which collided.
    tile: int
        The index of the tile with which the collision happened.
    position: tuple of int
            The position of the tile with which the collision happened.
    direction: int
        The direction of the collision relative to the entity.

    Methods
    -------
    default_handler_collision_tile(event)
        The default collision handler.
    """

    def __init__(self, tick: int, entity: Entity, tile: int, position: tuple, direction: int):
        """
        Initializes the CollisionWithTileEvent.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        entity: Entity
            The entity which collided.
        tile: int
            The index of the tile with which the collision happened.
        position: tuple of int
            The position of the tile with which the collision happened.
        direction: int
            The direction of the collision relative to the entity.
        """

        CollisionEvent.__init__(self, tick, entity, direction)

        self.tile = tile
        self.position = position

    @staticmethod
    def default_handler_collision_tile(event: "CollisionWithTileEvent") -> None:
        """
        The default collision handler.

        This handler is a default collision handler. It simply sets the speed of the entity to 0 along the event
        direction.

        Parameters
        ----------
        event: CollisionWithTileEvent
            The collision event fired.
        """

        direction = event.direction

        if direction == Direction.DIRECTION_EAST or direction == Direction.DIRECTION_WEST:
            event.entity.speed[0] = 0
        elif direction == Direction.DIRECTION_NORTH or direction == Direction.DIRECTION_SOUTH:
            event.entity.speed[1] = 0


class CollisionWithEntityEvent(CollisionEvent):
    """
    The type of CollisionEvent used for collisions between two entities.

    This kind of event is fired whenever two entities collide. Note that after being handled, the speed of the entities
    who collided must have changed to avoid colliding again (or else you are likely to be stuck in an infinite loop).
    This event can be reversed, meaning that both entities can swap their role.

    Attributes
    ----------
    tick: int
        The tick at which the event got fired.
    entity: Entity
        The entity which collided.
    other: entity
        The other entity with which the collision happened.
    direction: int
        The direction of the collision relative to the first entity.

    Methods
    -------
    reverse()
        Reverses the event by swapping the role of the entities.
    default_handler_collision_entity(event)
        The default collision handler.
    """

    def __init__(self, tick: int, entity: Entity, other: Entity, direction: int):
        """
        Initializes the CollisionWithEntityEvent.

        Parameters
        ----------
        tick: int
            The tick at which the event got fired.
        entity: Entity
            The entity which collided.
        other: entity
            The other entity with which the collision happened.
        direction: int
            The direction of the collision relative to the first entity.
        """

        CollisionEvent.__init__(self, tick, entity, direction)

        self.other = other

    def reverse(self) -> None:
        """
        Reverses the event by swapping the role of the entities.
        """

        self.entity, self.other = self.other, self.entity
        self.direction = Direction.opposite(self.direction)

    @staticmethod
    def default_handler_collision_entity(event: "CollisionWithEntityEvent") -> None:
        """
        The default collision handler.

        This handler is a default collision handler. It simply sets the speed of both entities to 0 along the event
        direction.

        Parameters
        ----------
        event: CollisionWithEntityEvent
            The collision event fired.
        """

        direction = event.direction

        axis = 0

        if direction == Direction.DIRECTION_NORTH or direction == Direction.DIRECTION_SOUTH:
            axis = 1

        if event.entity.speed[axis] * event.other.speed[axis] <= 0:
            event.entity.speed[axis] = 0
            event.other.speed[axis] = 0
        elif abs(event.entity.speed[axis]) >= abs(event.other.speed[axis]):
            event.entity.speed[axis] = 0
        else:
            event.other.speed[axis] = 0


class MovableSegment:
    """
    SAT util object used for 1D collisions.

    This class is used to apply the Separating Axis Theorem. A moving AABB should have its bounds projected on both
    plane axis as a movable segment. It allows to compute the time of impact or the time of separation of two objects.

    Attributes
    ----------
    low: float
        The lowest position of the segment.
    high: float
        The highest position of the segment.
    speed: float
        The speed of the segment.

    Methods
    -------
    should_impact(other)
        Returns whether or not the segments will impact.
    should_separate(other)
        Returns whether or not the segments will separate.
    time_of_impact(other)
        Computes the time at which the two segments will collide.
    time_of_separation(other)
        Computes the time at which the two segments will separate.
    """

    def __init__(self, low: float, high: float, speed: float = 0.0):
        """
        Initializes the MovableSegment.

        Parameters
        ----------
        low: float
            The lowest position of the segment.
        high: float
            The highest position of the segment.
        speed: float, optional
            The speed of the segment.
        """

        self.low = low
        self.high = high
        self.speed = speed

    def __contains__(self, other: "MovableSegment") -> bool:
        """
        Returns the inner point test.

        Tests if the specified segment is completely embedded in the main one.

        Parameters
        ----------
        other: MovableSegment
            The other movable segment with which the test is performed.

        Returns
        -------
        result: bool
            True if the tested segment is in the main segment and False otherwise.
        """

        return self.high >= other.low and self.low <= other.high

    def __eq__(self, other: "MovableSegment") -> bool:
        """
        Returns the equality test.

        Tests if the specified segment is strictly equals to the main one in terms of position and size.

        Parameters
        ----------
        other: MovableSegment
            The other movable segment with which the test is performed.

        Returns
        -------
        result: bool
            True if the tested segment is equal to the main segment and False otherwise.
        """

        return self.high == other.low or self.low == other.high

    def should_impact(self, other: "MovableSegment") -> bool:
        """
        Returns whether or not the segments will impact.

        Parameters
        ----------
        other: MovableSegment
            The other segment with which the test is performed.

        Returns
        -------
        result: bool
            True if the tested segment should impact with the main one and False otherwise.
        """

        return self.high < other.low and self.speed > other.speed or self.low > other.high and self.speed < other.speed

    def should_separate(self, other: "MovableSegment") -> bool:
        """
        Returns whether or not the segments will separate.

        Parameters
        ----------
        other: MovableSegment
            The other segment with which the test is performed.

        Returns
        -------
        result: bool
            True if the tested segment should separate from the main one and False otherwise.
        """

        return self.speed != other.speed

    def time_of_impact(self, other: "MovableSegment") -> float:
        """
        Computes the time at which the two segments will collide.

        This function should be called only if the segments will impact. It expresses the time of impact (TOI) between
        the two segment in fraction of tick.

        Parameters
        ----------
        other: MovableSegment
            The other segment.

        Returns
        -------
        time_of_impact: float
            The time of impact expressed in a fraction of tick.
        """

        if self.speed == other.speed:
            return -1

        if self.high < other.low:
            return (other.low - self.high) / (self.speed - other.speed)

        return (self.low - other.high) / (other.speed - self.speed)

    def time_of_separation(self, other: "MovableSegment") -> float:
        """
        Computes the time at which the two segments will separate.

        This function should be called only if the segments will separate. It expresses the time of separation (TOS)
        between the two segment in fraction of tick.

        Parameters
        ----------
        other: MovableSegment
            The other segment.

        Returns
        -------
        time_of_impact: float
            The time of separation expressed in a fraction of tick.
        """

        if self.speed > other.speed:
            return (other.high - self.low) / (self.speed - other.speed)

        return (self.high - other.low) / (other.speed - self.speed)


class CollisionObject:
    """
    Standard collision object representation.

    This object is used to create an abstract and logic pure representation of an entity. It uses the float
    representation of the bounding boxes for better precision when the local time is non-zero. It is useful when the
    entity are overridden by the user with inadequate data such as ctypes that cannot be used for parallelization. It
    also holds the index of the entity for simpler list usage.

    Attributes
    ----------
    index: int
        The index of the object corresponding to the entity from the world object list.
    bounding_box: AxisAlignedBoundingBox
        The AABB of the entity in float representation.
    bounding_box_expanded: AxisAlignedBoundingBox
        The minimal AABB covering the amount space in which the object moves.
    speed: numpy.ndarray
        The speed vector of the entity expressed in unit per tick.
    entity_type: type
        The type of the entity used as collider type.
    collides_with_tiles: bool
        Enables the collision detection with tiles.
    local_time: float
        The local time of the object within the current tick. This should be between 0 and 1.

    Methods
    -------
    should_collide_with(other)
        Returns whether or not the collision object should collide with the other.
    """

    def __init__(self, index: int, entity: Entity, local_time: float):
        """
        Initializes the CollisionObject.

        Parameters
        ----------
        index: int
            The index of the object corresponding to the entity from the world object list.
        entity: Entity
            The entity from which the collision object representation is created.
        local_time: float
            The local time of the object within the current tick. This should be between 0 and 1.
        """

        self.index = index

        self.bounding_box = entity.bounding_box.as_type(numpy.float32)
        self.bounding_box_expanded = self.bounding_box.expand(entity.speed * (1 - local_time))
        self.speed = entity.speed

        self.entity_type = type(entity)
        self._colliders = entity.colliders

        self.collides_with_tiles = entity.collides_with_tiles

        self.local_time = local_time

    def should_collide_with(self, other: "CollisionObject") -> bool:
        """
        Returns whether or not the collision object should collide with the other.

        Parameters
        ----------
        other: CollisionObject
            The other collision object with which the test is performed.
        Returns
        -------
        result: bool
            True if the object should collide with the other and False otherwise.
        """

        cls = other.entity_type

        for collider in self._colliders:
            if issubclass(cls, collider):
                return True

        return False


class QuadTree:
    """
    Spatial partitioning for collision detection optimization.

    This class holds the current object of the world with associated to the amount of space they move in. This is based
    on a structure of quad tree for 2D space partitioning adapted for AABBs, and speeds up the detection of potential
    collision pairs.

    Methods
    -------
    insert(collision_object)
        Inserts a new collision object in the tree.
    intersect(bounds, results=None, unique=None)
        Returns the list of object intersecting with the bounds.
    """

    DEFAULT_NODE_CAPACITY = 32
    DEFAULT_MAX_DEPTH = 5

    def __init__(self, bounds: AxisAlignedBoundingBox, node_capacity: int = DEFAULT_NODE_CAPACITY,
                 max_depth: int = DEFAULT_MAX_DEPTH):
        """
        Initializes the QuadTree.

        Parameters
        ----------
        bounds: AxisAlignedBoundingBox
            The bounds of the leaf.
        node_capacity: int, optional
            The object capacity of the leaf before it divides into smaller leaves.
        max_depth: int, optional
            The maximum depth of the tree.
        """

        self._nodes = []
        self._children = []

        self._bounds = bounds
        self._center = self._bounds.position + self._bounds.bounds / 2

        self._max_items = node_capacity
        self._max_depth = max_depth

    def _split(self) -> None:
        """
        Divides the leaf into four sub leaves.

        The leaf is divided in four other leaves of the same size (one for each corner). Each object of the leaf is
        placed into the sub leaves if it is possible.
        """

        width_low = self._bounds.bounds[0] // 2
        width_high = self._bounds.bounds[0] - width_low

        height_low = self._bounds.bounds[1] // 2
        height_high = self._bounds.bounds[1] - height_low

        self._children.append(QuadTree(AxisAlignedBoundingBox(
            self._bounds.position, (width_low, height_low)
        ), node_capacity=self._max_items, max_depth=self._max_depth - 1))

        self._children.append(QuadTree(AxisAlignedBoundingBox(
            self._bounds.position + (0, height_low), (width_low, height_high)
        ), node_capacity=self._max_items, max_depth=self._max_depth - 1))

        self._children.append(QuadTree(AxisAlignedBoundingBox(
            self._bounds.position + (width_low, height_low), (width_high, height_high)
        ), node_capacity=self._max_items, max_depth=self._max_depth - 1))

        self._children.append(QuadTree(AxisAlignedBoundingBox(
            self._bounds.position + (width_low, 0), (width_high, height_low)
        ), node_capacity=self._max_items, max_depth=self._max_depth - 1))

        nodes = self._nodes

        self._nodes = []

        for node in nodes:
            self._insert_into_children(node)

    def _insert_into_children(self, collision_object: CollisionObject) -> None:
        """
        Inserts the node into one of the children.

        Parameters
        ----------
        collision_object: CollisionObject
            The object to insert in the tree.
        """

        if self._center in collision_object.bounding_box_expanded:
            self._nodes.append(collision_object)
        else:
            if collision_object.bounding_box_expanded.position[0] <= self._center[0]:
                if collision_object.bounding_box_expanded.position[1] <= self._center[1]:
                    self._children[0].insert(collision_object)
                if collision_object.bounding_box_expanded.position[1] + \
                        collision_object.bounding_box_expanded.bounds[1] >= self._center[1]:
                    self._children[1].insert(collision_object)
            if collision_object.bounding_box_expanded.position[0] + \
                    collision_object.bounding_box_expanded.bounds[0] > self._center[0]:
                if collision_object.bounding_box_expanded.position[1] <= self._center[1]:
                    self._children[2].insert(collision_object)
                if collision_object.bounding_box_expanded.position[1] + \
                        collision_object.bounding_box_expanded.bounds[1] >= self._center[1]:
                    self._children[3].insert(collision_object)

    def insert(self, collision_object: CollisionObject) -> None:
        """
        Inserts a new collision object in the tree.

        Tries to insert a new collision object. If the leaf is already split, the object will be inserted in the sub
        leaves if it is possible.

        Parameters
        ----------
        collision_object: CollisionObject
            The object to insert in the tree.
        """

        if len(self._children) == 0:
            self._nodes.append(collision_object)

            if len(self._nodes) > self._max_items and self._max_depth > 0:
                self._split()
        else:
            self._insert_into_children(collision_object)

    def intersect(self, bounds: AxisAlignedBoundingBox, results: list = None, unique: set = None) -> list:
        """
        Returns the list of object intersecting with the bounds.

        Parameters
        ----------
        bounds: AxisAlignedBoundingBox
            The AABB of research.
        results: list of CollisionObject, optional
            The current result list. Leave it to None by default.
        unique: set of indexes, optional
            The current set of indexes of object in the result list.

        Returns
        -------
        results: list of CollisionObject
            The list of collision objects intersecting with the AABB.
        """

        if results is None:
            results = []
            unique = set()

        if self._children:
            if bounds.position[0] <= self._center[0]:
                if bounds.position[1] <= self._center[1]:
                    self._children[0].intersect(bounds, results, unique)
                if bounds.position[1] + bounds.bounds[1] >= self._center[1]:
                    self._children[1].intersect(bounds, results, unique)
            if bounds.position[0] + bounds.bounds[0] >= self._center[0]:
                if bounds.position[1] <= self._center[1]:
                    self._children[2].intersect(bounds, results, unique)
                if bounds.position[1] + bounds.bounds[1] >= self._center[1]:
                    self._children[3].intersect(bounds, results, unique)

        for node in self._nodes:
            _id = node.index

            if _id not in unique and bounds.intersects(node.bounding_box_expanded):
                results.append(node)
                unique.add(_id)

        return results


class CollisionPseudoEvent:
    """
    A generic object to represent the collision events.

    Note that this object is not event, it merely serves as a placeholder. When multiple collision events occurs in the
    same tick, the set of pseudo event can be reduced by order of importance and independence. Two event are independent
    if their colliders set is disjoint. A event is more important than an other one if they are dependent and the time
    of impact of the first event is lesser than the one of the latter. After being sorted and separated, the pseudo
    events can be converted into standard collision event and being processed by the event queue.

    Attributes
    ----------
    time_of_impact: float
        The time of impact of the collision within the tick (between 0 and 1).
    colliders: list of int
        The indexes of the colliders, either the collision object index or the tile data.
    collision_type: int
        The type of collision, either collision vs tile or collision vs entity.
    collision_direction: int
        The direction of the collision relative to the first collider.
    """

    COLLISION_NONE = 0
    COLLISION_TILE = 1
    COLLISION_ENTITY = 2

    def __init__(self, time_of_impact: float, colliders: tuple, collision_type: int, collision_direction: int):
        """
        Initializes the CollisionPseudoEvent.

        Parameters
        ----------
        time_of_impact: float
            The time of impact of the collision within the tick (between 0 and 1).
        colliders: tuple of int
            The indexes of the colliders, either the collision object index or the tile data.
        collision_type: int
            The type of collision, either collision vs tile or collision vs entity.
        collision_direction: int
            The direction of the collision relative to the first collider.
        """

        self.time_of_impact = time_of_impact
        self.colliders = colliders
        self.collision_type = collision_type
        self.collision_direction = collision_direction

    def __contains__(self, other: "CollisionPseudoEvent") -> bool:
        """
        Returns if the other event is less important and dependent.

        Tests if the specified event is less important than the main while being dependent. It means that both events
        have at least a common collider and that the time of impact of the tested event is greater than the one of the
        latter.

        Parameters
        ----------
        other: CollisionPseudoEvent
            The other event with which the test is performed.

        Returns
        -------
        result: bool
            True if the tested event is less important and dependent and False otherwise.
        """

        if self.time_of_impact < other.time_of_impact:
            if self.collision_type == CollisionPseudoEvent.COLLISION_TILE:
                if other.collision_type == CollisionPseudoEvent.COLLISION_TILE:
                    if self.colliders[0] == other.colliders[0]:
                        return True
                elif other.collision_type == CollisionPseudoEvent.COLLISION_ENTITY:
                    if self.colliders[0] in other.colliders:
                        return True
            elif self.collision_type == CollisionPseudoEvent.COLLISION_ENTITY:
                if other.collision_type == CollisionPseudoEvent.COLLISION_TILE:
                    if other.colliders[0] in self.colliders:
                        return True
                elif other.collision_type == CollisionPseudoEvent.COLLISION_ENTITY:
                    if self.colliders[0] in other.colliders or self.colliders[1] in other.colliders:
                        return True

        return False

    def __eq__(self, other: "CollisionPseudoEvent") -> bool:
        """
        Returns if the events are equally important.

        Tests if the specified event is as important as the main one while being dependent. It means that both events
        have at least one common collider and their times of impact are equal.

        Parameters
        ----------
        other: any
            The other event with which the test is performed.

        Returns
        -------
        result: bool
            True if the tested event is as important and dependent and False otherwise.
        """

        if self.time_of_impact == other.time_of_impact:
            if self.collision_type == CollisionPseudoEvent.COLLISION_TILE:
                if other.collision_type == CollisionPseudoEvent.COLLISION_TILE:
                    if self.colliders[0] == other.colliders[0]:
                        return True
                elif other.collision_type == CollisionPseudoEvent.COLLISION_ENTITY:
                    if self.colliders[0] in other.colliders:
                        return True
            elif self.collision_type == CollisionPseudoEvent.COLLISION_ENTITY:
                if other.collision_type == CollisionPseudoEvent.COLLISION_TILE:
                    if other.colliders[0] in self.colliders:
                        return True
                elif other.collision_type == CollisionPseudoEvent.COLLISION_ENTITY:
                    if self.colliders[0] in other.colliders or self.colliders[1] in other.colliders:
                        return True

        return False

    def __hash__(self) -> int:
        """
        Returns the hash associated to the event.

        Returns
        -------
        hash: int
            The event hash.
        """

        return hash(self.colliders + (self.time_of_impact, self.collision_type, self.collision_type))


class WorldUpdater:
    """
    The collision processing core.

    The world logic is done here. It contains the necessary functions to find the next collision events within the
    current tick. Theses functions can be paralleled for better performance.

    Attributes
    ----------
    tiles: numpy.ndarray
        The tiles array of the level.
    logic_area: AxisAlignedBoundingBox
        The area over which the logic is performed.
    logic_tile: bool
        Enables the collision detection with tiles if set to True.
    logic_entity: bool
        Enables the collision detection with entities if set to True.

    Methods
    -------
    fetch_next_events(entities, local_times)
        Returns the next collision events to fire.
    """

    DEFAULT_ENTITY_PER_THREAD = 32

    def __init__(self, tile_manager: TileManager, tiles: numpy.ndarray, logic_area: AxisAlignedBoundingBox = None,
                 logic_tile: bool = True, logic_entity: bool = True, multi_threading: bool = True,
                 entity_per_thread: int = DEFAULT_ENTITY_PER_THREAD,
                 node_capacity: int = QuadTree.DEFAULT_NODE_CAPACITY, max_depth: int = QuadTree.DEFAULT_MAX_DEPTH):
        """
        Initializes the WorldUpdater.

        Parameters
        ----------
        tile_manager: TileManager
            The tile manager containing the tile data.
        tiles: numpy.ndarray
            The tiles array of the level.
        logic_area: AxisAlignedBoundingBox, optional
            The area over which the logic is performed.
        logic_tile: bool, optional
            Enables the collision detection with the tiles if set to True.
        logic_entity: bool, optional
            Enables the collision detection with the entities is set to True.
        multi_threading: bool, optional
            Enables the multi-threading mode if set to True.
        entity_per_thread: int, optional
            The number of entity in each thread.
        node_capacity: int, optional
            The object capacity of the leaf before it divides into smaller leaves.
        max_depth: int, optional
            The maximum depth of the tree.
        """

        self._tile_manager = tile_manager
        self.tiles = tiles

        if logic_area is None:
            logic_area = AxisAlignedBoundingBox((0, 0), numpy.array(tiles.shape) * tile_manager.tile_size)

        self.logic_area = logic_area

        self.logic_tile = logic_tile
        self.logic_entity = logic_entity

        self._multi_threading = multi_threading
        self._entity_per_thread = entity_per_thread

        self._node_capacity = node_capacity
        self._max_depth = max_depth

    def fetch_next_events(self, entities: list, local_times: list) -> list:
        """
        Returns the next collision events to fire.

        Finds the next maximal independent events to fire. It means that multiple collision events can be solved in a
        single tick as long as they are independent from each other. This functions uses multiple threads if the
        multi-threading mode is enabled.

        Parameters
        ----------
        entities: list of Entity
            The list of entities of the world.
        local_times: list of float
            The corresponding of local time for each entities.

        Returns
        -------
        events: list of CollisionPseudoEvent
            The list of independent events to fire.
        """

        collision_objects = []

        events = {}

        tree = QuadTree(self.logic_area, node_capacity=32, max_depth=5)

        for index in range(len(entities)):
            collision_object = CollisionObject(index, entities[index], local_times[index])

            tree.insert(collision_object)
            collision_objects.append(collision_object)

            events[collision_object] = None

        if self._multi_threading:
            splits = [collision_objects[x * self._entity_per_thread:self._entity_per_thread * (x + 1)]
                      for x in range(len(collision_objects) // self._entity_per_thread +
                                     (len(collision_objects) % self._entity_per_thread > 0))]

            with ThreadPool(processes=len(splits)) as pool:
                results = pool.starmap(self._process_collision_events, zip(splits, repeat(tree)))

            events = [y for x in results for y in x]
        else:
            events = self._process_collision_events(collision_objects, tree)

        next_events = []

        for event in events:
            if event in next_events:
                continue

            for other in events:
                if event != other and event in other:
                    break
            else:
                next_events.append(event)

        return next_events

    def _process_collision_events(self, entities_colliding: list, tree: QuadTree) -> list:
        """
        Finds every next possible events.

        This function should be called on a subset of entities. Note that it returns every potential events, some may
        not be independent and thus have to be sorted by size before being solved. For instance a single object can
        issue several events with other objects, thus the order of resolution is important.

        Parameters
        ----------
        entities_colliding: list of CollisionObject
            The subset of collision objects over which the events will be searched.
        tree: QuadTree
            The quad tree spatial partition containing the other collision objects.

        Returns
        -------
        events: list of CollisionPseudoEvent
            The list of potential events to fire.
        """

        events = []

        for entity_colliding in entities_colliding:
            time_of_impact = 1.0

            colliders = []
            collision_type = CollisionPseudoEvent.COLLISION_NONE
            collision_direction = Direction.DIRECTION_NONE

            event_found = False

            others = tree.intersect(entity_colliding.bounding_box_expanded)

            for collided in others:
                if self.logic_entity and entity_colliding.index < collided.index and \
                        entity_colliding.should_collide_with(collided) and \
                        collided.should_collide_with(entity_colliding):

                    potential_time_of_impact, direction = WorldUpdater._apply_sat(entity_colliding, collided)

                    if direction != Direction.DIRECTION_NONE and potential_time_of_impact < time_of_impact:
                        time_of_impact = potential_time_of_impact
                        colliders = (entity_colliding.index, collided.index)
                        collision_type = CollisionPseudoEvent.COLLISION_ENTITY
                        collision_direction = direction

                        event_found = True

            if self.logic_tile and entity_colliding.collides_with_tiles:
                potential_time_of_impact, tile, x, y, direction = self._tile_collision_detection(entity_colliding)

                if direction != Direction.DIRECTION_NONE and potential_time_of_impact < time_of_impact:
                    time_of_impact = potential_time_of_impact
                    colliders = (entity_colliding.index, tile, x, y)
                    collision_type = CollisionPseudoEvent.COLLISION_TILE
                    collision_direction = direction

                    event_found = True

            if event_found:
                events.append(CollisionPseudoEvent(time_of_impact, colliders, collision_type, collision_direction))

        return events

    @staticmethod
    def _apply_sat(collider: CollisionObject, collided: CollisionObject) -> (float, int):
        """
        Applies the SAT to two collision objects.

        Applies the Separation Axis Theorem to two collision objects and determines whether or not they will collide and
        if so, it will compute the time of impact between both object within the current tick. It uses the float
        representation of the AABBs for better precision.

        Parameters
        ----------
        collider: CollisionObject
            The collision object associated to the collider.
        collided: CollisionObject
            The collision object associated to the collided.

        Returns
        -------
        time_of_impact, direction: float, int
            The time of impact and the direction of the collision if there is any.
        """

        if collider.local_time < collided.local_time:
            time_of_impact, direction = WorldUpdater._apply_sat(collided, collider)

            return time_of_impact, Direction.opposite(direction)

        collider_bounds = collider.bounding_box.copy()
        collided_bounds = collided.bounding_box.copy()

        collided_bounds.position += collided.speed * (collider.local_time - collided.local_time)

        maximum_time_of_impact = 0
        minimum_time_of_separation = 1 - collider.local_time

        collision_direction = Direction.DIRECTION_NONE

        relative_speed = collider.speed - collided.speed

        for axis in range(2):
            collider_segment = MovableSegment(
                collider_bounds.position[axis],
                collider_bounds.position[axis] + collider_bounds.bounds[axis],
                collider.speed[axis]
            )

            collided_segment = MovableSegment(
                collided_bounds.position[axis],
                collided_bounds.position[axis] + collided_bounds.bounds[axis],
                collided.speed[axis]
            )

            if collider_segment not in collided_segment:
                if collider_segment.should_impact(collided_segment):
                    time_of_impact = collider_segment.time_of_impact(collided_segment)

                    if time_of_impact > maximum_time_of_impact:
                        maximum_time_of_impact = time_of_impact

                        if relative_speed[axis] > 0:
                            collision_direction = Direction.DIRECTION_EAST
                        elif relative_speed[axis] < 0:
                            collision_direction = Direction.DIRECTION_WEST
                        else:
                            continue

                        if axis != 0:
                            collision_direction = Direction.next_left(collision_direction)

                    if collider_segment.should_separate(collided_segment):
                        time_of_separation = collider_segment.time_of_separation(collided_segment)

                        if time_of_separation < minimum_time_of_separation:
                            minimum_time_of_separation = time_of_separation

                else:
                    return 1.0, Direction.DIRECTION_NONE

            if collider_segment == collided_segment:
                if collider_segment.should_separate(collided_segment):
                    if maximum_time_of_impact == 0:
                        if relative_speed[axis] > 0:
                            collision_direction = Direction.DIRECTION_EAST
                        elif relative_speed[axis] < 0:
                            collision_direction = Direction.DIRECTION_WEST
                        else:
                            continue

                        if axis != 0:
                            collision_direction = Direction.next_left(collision_direction)

                        time_of_separation = collider_segment.time_of_separation(collided_segment)

                        if time_of_separation < minimum_time_of_separation:
                            minimum_time_of_separation = time_of_separation

        if minimum_time_of_separation > maximum_time_of_impact and maximum_time_of_impact < 1 - collider.local_time:
            return collider.local_time + maximum_time_of_impact, collision_direction

        return 1.0, Direction.DIRECTION_NONE

    def _tile_collision_detection(self, collider: CollisionObject) -> (float, int, int, int, int):
        """
        Applies the SAT between a collision object and the tiles.

        Applies a modified version of the Separation Axis Theorem for static objects. This determines whether or not the
        collision object will collide with a tile from the terrain and if so, it will compute the time of impact between
        the object and the tile within the current tick.

        Parameters
        ----------
        collider: CollisionObject
            The collision object associated to the collider.

        Returns
        -------
        time_of_impact, tile, direction: float, int, int, int, int
            The time of impact, the tile index and its position and the direction of the collision if there is any.
        """

        bounds = collider.bounding_box.copy()

        speed = collider.speed
        local_time = collider.local_time

        pre_min = bounds.position.astype(numpy.int32)
        pre_max = (bounds.position + bounds.bounds).astype(numpy.int32)
        post_min = (pre_min + speed * (1 - local_time)).astype(numpy.int32)
        post_max = (pre_max + speed * (1 - local_time)).astype(numpy.int32)

        pre_min_tile = pre_min // self._tile_manager.tile_size
        pre_max_tile = pre_max // self._tile_manager.tile_size
        post_min_tile = post_min // self._tile_manager.tile_size
        post_max_tile = post_max // self._tile_manager.tile_size

        if pre_max[0] % self._tile_manager.tile_size == 0:
            pre_max_tile[0] -= 1

        if pre_max[1] % self._tile_manager.tile_size == 0:
            pre_max_tile[1] -= 1

        if post_max[0] % self._tile_manager.tile_size == 0:
            post_max_tile[0] -= 1

        if post_max[1] % self._tile_manager.tile_size == 0:
            post_max_tile[1] -= 1

        if pre_min_tile[0] == post_min_tile[0] and pre_min_tile[1] == post_min_tile[1] and \
                pre_max_tile[0] == post_max_tile[0] and pre_max_tile[1] == post_max_tile[1]:
            return 1.0, -1, -1, -1, Direction.DIRECTION_NONE

        if post_min_tile[1] == pre_min_tile[1] and post_max_tile[1] == pre_max_tile[1]:
            if post_max_tile[0] > pre_max_tile[0]:
                for x in range(pre_max_tile[0] + 1, post_max_tile[0] + 1):
                    for y in range(pre_min_tile[1], pre_max_tile[1] + 1):
                        tile = self._get_tile_at(x, y)

                        if tile > 0:
                            collision_map = self._tile_manager.get_collision_map(tile)

                            if collision_map.collide_west:
                                return (x * self._tile_manager.tile_size - pre_max[0]) / speed[0] + local_time, tile, \
                                       x, y, Direction.DIRECTION_EAST
                return 1.0, -1, -1, -1, Direction.DIRECTION_NONE

            elif post_min_tile[0] < pre_min_tile[0]:
                for x in range(pre_min_tile[0] - 1, post_min_tile[0] - 1, -1):
                    for y in range(pre_min_tile[1], pre_max_tile[1] + 1):
                        tile = self._get_tile_at(x, y)

                        if tile > 0:
                            collision_map = self._tile_manager.get_collision_map(tile)

                            if collision_map.collide_east:
                                return ((x + 1) * self._tile_manager.tile_size - pre_min[0]) / speed[0] + local_time, \
                                       tile, x, y, Direction.DIRECTION_WEST
                return 1.0, -1, -1, -1, Direction.DIRECTION_NONE

        elif post_min_tile[0] == pre_min_tile[0] and post_max_tile[0] == pre_max_tile[0]:
            if post_max_tile[1] > pre_max_tile[1]:
                for y in range(pre_max_tile[1] + 1, post_max_tile[1] + 1):
                    for x in range(pre_min_tile[0], pre_max_tile[0] + 1):
                        tile = self._get_tile_at(x, y)

                        if tile > 0:
                            collision_map = self._tile_manager.get_collision_map(tile)

                            if collision_map.collide_south:
                                return (y * self._tile_manager.tile_size - pre_max[1]) / speed[1] + local_time, tile, \
                                       x, y, Direction.DIRECTION_NORTH
                return 1.0, -1, -1, -1, Direction.DIRECTION_NONE

            elif post_min_tile[1] < pre_min_tile[1]:
                for y in range(pre_min_tile[1] - 1, post_min_tile[1] - 1, -1):
                    for x in range(pre_min_tile[0], pre_max_tile[0] + 1):
                        tile = self._get_tile_at(x, y)

                        if tile > 0:
                            collision_map = self._tile_manager.get_collision_map(tile)

                            if collision_map.collide_north:
                                return ((y + 1) * self._tile_manager.tile_size - pre_min[1]) / speed[1] + local_time, \
                                       tile, x, y, Direction.DIRECTION_SOUTH
                return 1.0, -1, -1, -1, Direction.DIRECTION_NONE

        if speed[0] > 0:
            if speed[1] > 0:
                vertices = [(pre_min[0], pre_min[1]), (pre_max[0] + 1, pre_min[1]),
                            (post_max[0] + 1, post_min[1]), (post_max[0] + 1, post_max[1] + 1),
                            (post_min[0], post_max[1] + 1), (pre_min[0], pre_max[1] + 1)]
            else:
                vertices = [(pre_min[0], pre_max[1] + 1), (pre_min[0], pre_min[1]),
                            (post_min[0], post_min[1]), (post_max[0] + 1, post_min[1]),
                            (post_max[0] + 1, post_max[1] + 1), (pre_max[0] + 1, pre_max[1] + 1)]

        else:
            if speed[1] > 0:
                vertices = [(pre_max[0] + 1, pre_min[1]), (pre_max[0] + 1, pre_max[1] + 1),
                            (post_max[0] + 1, post_max[1] + 1), (post_min[0], post_max[1] + 1),
                            (post_min[0], post_min[1]), (pre_min[0], pre_min[1])]
            else:
                vertices = [(pre_max[0] + 1, pre_max[1] + 1), (pre_min[0], pre_max[1] + 1),
                            (post_min[0], post_max[1] + 1), (post_min[0], post_min[1]),
                            (post_max[0] + 1, post_min[1]), (pre_max[0] + 1, pre_min[1])]

        time_of_impact = 1 - local_time
        collision_tile = 0
        collision_tile_x = 0
        collision_tile_y = 0
        collision_direction = Direction.DIRECTION_NONE

        found_x = False
        found_y = False

        min_tile = numpy.min((pre_min_tile, post_min_tile), axis=0)
        max_tile = numpy.max((pre_max_tile, post_max_tile), axis=0)

        raster = numpy.zeros(max_tile - min_tile + 1, dtype=bool)

        indexes_x = list(range(0, self._tile_manager.tile_size, int(bounds.bounds[0])))
        indexes_y = list(range(0, self._tile_manager.tile_size, int(bounds.bounds[1])))

        if self._tile_manager.tile_size - 1 not in indexes_x:
            indexes_x.append(self._tile_manager.tile_size - 1)

        if self._tile_manager.tile_size - 1 not in indexes_y:
            indexes_y.append(self._tile_manager.tile_size - 1)

        for x_tile in range(0, max_tile[0] - min_tile[0] + 1):
            for y_tile in range(0, max_tile[1] - min_tile[1] + 1):
                x = (x_tile + min_tile[0]) * self._tile_manager.tile_size
                y = (y_tile + min_tile[1]) * self._tile_manager.tile_size

                for offset in indexes_x:
                    if WorldUpdater._inner_polygon_test(vertices, x + offset, y) or \
                            WorldUpdater._inner_polygon_test(vertices, x + offset, y + self._tile_manager.tile_size):
                        raster[x_tile, y_tile] = True

                        break
                else:
                    for offset in indexes_y:
                        if WorldUpdater._inner_polygon_test(vertices, x + self._tile_manager.tile_size, y + offset) or \
                                WorldUpdater._inner_polygon_test(vertices, x, y + offset):
                            raster[x_tile, y_tile] = True

                            break

        if speed[0] > 0:
            if speed[1] > 0:
                for x in range(pre_min_tile[0], post_max_tile[0] + 1):
                    for y in range(pre_min_tile[1], post_max_tile[1] + 1):
                        if pre_min_tile[0] <= x <= pre_max_tile[0] and pre_min_tile[1] <= y <= pre_max_tile[1] or \
                                not raster[x - min_tile[0], y - min_tile[1]]:
                            continue

                        tile = self._get_tile_at(x, y)

                        if tile > 0:
                            collision_map = self._tile_manager.get_collision_map(tile)

                            times_of_impact = \
                                (x * self._tile_manager.tile_size - pre_max[0]) / speed[0], \
                                (y * self._tile_manager.tile_size - pre_max[1]) / speed[1]

                            if times_of_impact[0] >= times_of_impact[1]:
                                if collision_map.collide_west and times_of_impact[0] < time_of_impact:
                                    time_of_impact = times_of_impact[0]

                                    if found_y:
                                        return time_of_impact + local_time, tile, x, y, Direction.DIRECTION_EAST

                                    collision_tile = tile
                                    collision_tile_x = x
                                    collision_tile_y = y
                                    collision_direction = Direction.DIRECTION_EAST

                                    found_x = True
                            else:
                                if collision_map.collide_south and times_of_impact[1] < time_of_impact:
                                    time_of_impact = times_of_impact[1]

                                    if found_x:
                                        return time_of_impact + local_time, tile, x, y, Direction.DIRECTION_NORTH

                                    collision_tile = tile
                                    collision_tile_x = x
                                    collision_tile_y = y
                                    collision_direction = Direction.DIRECTION_NORTH

                                    found_y = True

            else:
                for x in range(pre_min_tile[0], post_max_tile[0] + 1):
                    for y in range(pre_max_tile[1], post_min_tile[1] - 1, -1):
                        if pre_min_tile[0] <= x <= pre_max_tile[0] and pre_min_tile[1] <= y <= pre_max_tile[1] or \
                                not raster[x - min_tile[0], y - min_tile[1]]:
                            continue

                        tile = self._get_tile_at(x, y)

                        if tile > 0:
                            collision_map = self._tile_manager.get_collision_map(tile)

                            times_of_impact = \
                                (x * self._tile_manager.tile_size - pre_max[0]) / speed[0], \
                                ((y + 1) * self._tile_manager.tile_size - pre_min[1]) / speed[1]

                            if times_of_impact[0] >= times_of_impact[1]:
                                if collision_map.collide_west and times_of_impact[0] < time_of_impact:
                                    time_of_impact = times_of_impact[0]

                                    if found_y:
                                        return time_of_impact + local_time, tile, x, y, Direction.DIRECTION_EAST

                                    collision_tile = tile
                                    collision_tile_x = x
                                    collision_tile_y = y
                                    collision_direction = Direction.DIRECTION_EAST

                                    found_x = True
                            else:
                                if collision_map.collide_north and times_of_impact[1] < time_of_impact:
                                    time_of_impact = times_of_impact[1]

                                    if found_x:
                                        return time_of_impact + local_time, tile, x, y, Direction.DIRECTION_SOUTH

                                    collision_tile = tile
                                    collision_tile_x = x
                                    collision_tile_y = y
                                    collision_direction = Direction.DIRECTION_SOUTH

                                    found_y = True

        else:
            if speed[1] > 0:
                for x in range(pre_max_tile[0], post_min_tile[0] - 1, -1):
                    for y in range(pre_min_tile[1], post_max_tile[1] + 1):
                        if pre_min_tile[0] <= x <= pre_max_tile[0] and pre_min_tile[1] <= y <= pre_max_tile[1] or \
                                not raster[x - min_tile[0], y - min_tile[1]]:
                            continue

                        tile = self._get_tile_at(x, y)

                        if tile > 0:
                            collision_map = self._tile_manager.get_collision_map(tile)

                            times_of_impact = \
                                ((x + 1) * self._tile_manager.tile_size - pre_min[0]) / speed[0], \
                                (y * self._tile_manager.tile_size - pre_max[1]) / speed[1]

                            if times_of_impact[0] >= times_of_impact[1]:
                                if collision_map.collide_east and times_of_impact[0] < time_of_impact:
                                    time_of_impact = times_of_impact[0]

                                    if found_y:
                                        return time_of_impact + local_time, tile, x, y, Direction.DIRECTION_WEST

                                    collision_tile = tile
                                    collision_tile_x = x
                                    collision_tile_y = y
                                    collision_direction = Direction.DIRECTION_WEST

                                    found_x = True
                            else:
                                if collision_map.collide_south and times_of_impact[1] < time_of_impact:
                                    time_of_impact = times_of_impact[1]

                                    if found_x:
                                        return time_of_impact + local_time, tile, x, y, Direction.DIRECTION_NORTH

                                    collision_tile = tile
                                    collision_tile_x = x
                                    collision_tile_y = y
                                    collision_direction = Direction.DIRECTION_NORTH

                                    found_y = True

            else:
                for x in range(pre_max_tile[0], post_min_tile[0] - 1, -1):
                    for y in range(pre_max_tile[1], post_min_tile[1] - 1, -1):
                        if pre_min_tile[0] <= x <= pre_max_tile[0] and pre_min_tile[1] <= y <= pre_max_tile[1] or \
                                not raster[x - min_tile[0], y - min_tile[1]]:
                            continue

                        tile = self._get_tile_at(x, y)

                        if tile > 0:
                            collision_map = self._tile_manager.get_collision_map(tile)

                            times_of_impact = \
                                ((x + 1) * self._tile_manager.tile_size - pre_min[0]) / speed[0], \
                                ((y + 1) * self._tile_manager.tile_size - pre_min[1]) / speed[1]

                            if times_of_impact[0] >= times_of_impact[1]:
                                if collision_map.collide_east and times_of_impact[0] < time_of_impact:
                                    time_of_impact = times_of_impact[0]

                                    if found_y:
                                        return time_of_impact + local_time, tile, x, y, Direction.DIRECTION_WEST

                                    collision_tile = tile
                                    collision_tile_x = x
                                    collision_tile_y = y
                                    collision_direction = Direction.DIRECTION_WEST

                                    found_x = True
                            else:
                                if collision_map.collide_north and times_of_impact[1] < time_of_impact:
                                    time_of_impact = times_of_impact[1]

                                    if found_x:
                                        return time_of_impact + local_time, tile, x, y, Direction.DIRECTION_SOUTH

                                    collision_tile = tile
                                    collision_tile_x = x
                                    collision_tile_y = y
                                    collision_direction = Direction.DIRECTION_SOUTH

                                    found_y = True

        if found_x or found_y:
            return time_of_impact + local_time, collision_tile, collision_tile_x, collision_tile_y, collision_direction

        return 1.0, -1, -1, -1, Direction.DIRECTION_NONE

    @staticmethod
    def _inner_polygon_test(vertices: list, x: int, y: int) -> bool:
        """
        Returns the inner point test for a polygon.

        This function is used to determine the tiles with which the object is likely to go through. This is used to
        raster the tile space under the sweep of the object.

        Parameters
        ----------
        vertices: list
            The list of vertices (couples of int) of the polygon to raster.
        x: int
            The x position of the point.
        y: int
            The y position of the point.

        Returns
        -------
        result: bool
            True if the tested point is in the polygon and False otherwise.
        """

        valid_positive = False
        valid_negative = False

        for vertex in range(len(vertices)):
            if ((vertices[vertex][1] > y) != (vertices[vertex - 1][1] > y)) and (
                    x < vertices[vertex][0] + (vertices[vertex - 1][0] - vertices[vertex][0]) * (
                    y - vertices[vertex][1]) / (vertices[vertex - 1][1] - vertices[vertex][1])):
                valid_positive = not valid_positive

            if ((vertices[vertex][1] < y + 1) != (vertices[vertex - 1][1] < y + 1)) and (
                    - x - 1 < - vertices[vertex][0] + (vertices[vertex][0] - vertices[vertex - 1][0]) * (
                    - y - 1 + vertices[vertex][1]) / (vertices[vertex][1] - vertices[vertex - 1][1])):
                valid_negative = not valid_negative

        return valid_positive or valid_negative

    def _get_tile_at(self, position_x: int, position_y: int) -> int:
        """
        Returns the index of the tile at the given position.

        Parameters
        ----------
        position_x: int
            The position of the tile along the x axis.
        position_y: int
            The position of the tile along the y axis.

        Returns
        -------
        id_tile: int
            The index of the tile.
        """

        if 0 <= position_x < self.tiles.shape[0] and 0 <= position_y < self.tiles.shape[1]:
            return self.tiles[position_x, position_y]

        return -1


class UnsolvedCollisionError(Exception):
    """
    An error raised when a collision is not solved.

    This kind of error is raised when a collision event is not solved, in other words when the same event occurs twice
    in the loop. This means that the event will be reprocessed indefinitely by the handlers. Note that the event
    handlers have to be deterministic regarding the update of the object, for instance, the speed vector of the
    colliding object should always be modified in a way that it does not instantly collides with the same other object.
    """


class World:
    """
    The main game data holder.

    This object contains the current played level with the tiles and the world objects. At each tick, the update
    function should be called to move the entities and fire the collision events.

    Attributes
    ----------
    tiles: numpy.ndarray
        The tiles array of the level.
    world_objects: list of WorldObject
        The list of world object spawned.
    background: str
        The background name.
    logic_area: AxisAlignedBoundingBox, optional
        The area over which the logic is performed.
    logic_tile: bool, optional
        Enables the collision detection with the tiles if set to True.
    logic_entity: bool, optional
        Enables the collision detection with the entities is set to True.

    Methods
    -------
    update(tick)
        Updates the world objects.
    spawn(world_object)
        Spawns a new world object.
    """

    def __init__(self, tile_manager: TileManager, event_queue: EventQueue, tiles: numpy.ndarray, background: str,
                 logic_area: AxisAlignedBoundingBox = None, logic_tile: bool = True, logic_entity: bool = True,
                 safe_mode: bool = True, multi_threading: bool = True,
                 entity_per_thread: int = WorldUpdater.DEFAULT_ENTITY_PER_THREAD,
                 node_capacity: int = QuadTree.DEFAULT_NODE_CAPACITY, max_depth: int = QuadTree.DEFAULT_MAX_DEPTH):
        """
        Initializes the World.

        Parameters
        ----------
        tile_manager: TileManager
            The tile manager containing the tile data.
        event_queue: EventQueue
            The main event queue used to handle events.
        tiles: numpy.ndarray
            The tiles array of the level.
        background: str
            The background name.
        logic_area: AxisAlignedBoundingBox, optional
            The area over which the logic is performed.
        logic_tile: bool, optional
            Enables the collision detection with the tiles if set to True.
        logic_entity: bool, optional
            Enables the collision detection with the entities is set to True.
        safe_mode: bool, optional
            Enables the update function safe mode (preventing infinite loop) if set to True.
        multi_threading: bool, optional
            Enables the multi-threading mode if set to True.
        entity_per_thread: int, optional
            The number of entity in each thread.
        node_capacity: int, optional
            The object capacity of the leaf before it divides into smaller leaves.
        max_depth: int, optional
            The maximum depth of the tree.
        """

        self._event_queue = event_queue

        self.background = background

        self.world_objects = []

        self._updater = WorldUpdater(
            tile_manager, tiles, logic_area, logic_tile, logic_entity, multi_threading, entity_per_thread,
            node_capacity, max_depth
        )

        self._safe_mode = safe_mode

    @property
    def tiles(self) -> numpy.ndarray:
        """
        The tiles property containing the tiles indexes of the level.
        """

        return self._updater.tiles

    @tiles.setter
    def tiles(self, tiles: numpy.ndarray) -> None:
        """
        Setter function for the tiles array.

        Parameters
        ----------
        tiles: numpy.ndarray
            The array of tile indexes of the level.
        """

        self._updater.tiles = tiles

    @property
    def logic_area(self) -> AxisAlignedBoundingBox:
        """
        The logic area property specifying the area over which the logic is performed.
        """

        return self._updater.logic_area

    @logic_area.setter
    def logic_area(self, logic_area: AxisAlignedBoundingBox) -> None:
        """
        Setter function for the logic area.

        Parameters
        ----------
        logic_area: AxisAlignedBoundingBox
            The AABB over which the logic is performed.
        """

        self._updater.logic_area = logic_area

    @property
    def logic_tile(self) -> bool:
        """
        The logic tile property specifying whether or not the collision detection should be done with tiles.
        """

        return self._updater.logic_tile

    @logic_tile.setter
    def logic_tile(self, logic_tile: bool) -> None:
        """
        Setter function for the logic tile flag.

        Parameters
        ----------
        logic_tile: bool
            Enables the collision detection with tiles if set to True.
        """

        self._updater.logic_tile = logic_tile

    @property
    def logic_entity(self) -> bool:
        """
        The logic entity property specifying whether or not the collision detection should be done with entities.
        """

        return self._updater.logic_entity

    @logic_entity.setter
    def logic_entity(self, logic_entity: bool) -> None:
        """
        Setter function for the logic entity flag.

        Parameters
        ----------
        logic_entity: bool
            Enables the collision detection with the entities if set to True.
        """

        self._updater.logic_entity = logic_entity

    def update(self, tick: int) -> None:
        """
        Updates the world objects.

        Fetches the collision and fires them using the world updater.

        Parameters
        ----------
        tick: int
            The current logic tick.

        Raises
        ------
        UnsolvedCollisionError
            If a same event is fired twice in a single tick.
        """

        to_destroy = []
        entities = []
        local_times = []

        past_events = set()

        for world_object in self.world_objects:
            if world_object.should_be_destroyed:
                to_destroy.append(world_object)
            else:
                if isinstance(world_object, Entity):
                    entities.append(world_object)
                    local_times.append(0.0)

        for world_object in to_destroy:
            self.world_objects.remove(world_object)

        if len(entities) == 0:
            return

        collision_remaining = True

        while collision_remaining:
            collision_remaining = False

            events = self._updater.fetch_next_events(entities, local_times)

            for event in events:
                if self._safe_mode and event in past_events:
                    raise UnsolvedCollisionError()

                if event.collision_type == CollisionPseudoEvent.COLLISION_ENTITY:
                    for collider in event.colliders:
                        entity = entities[collider]

                        position = entity.position + entity.speed * (event.time_of_impact - local_times[collider])
                        entity.position = numpy.floor(position)

                        local_times[collider] = event.time_of_impact

                    self._event_queue.fire_event(CollisionWithEntityEvent(
                            tick, entities[event.colliders[0]], entities[event.colliders[1]], event.collision_direction
                    ))

                if event.collision_type == CollisionPseudoEvent.COLLISION_TILE:
                    entity = entities[event.colliders[0]]

                    position = entity.position + entity.speed * (event.time_of_impact - local_times[event.colliders[0]])
                    entity.position = numpy.floor(position)

                    local_times[event.colliders[0]] = event.time_of_impact

                    tile_position = event.colliders[2:]

                    self._event_queue.fire_event(CollisionWithTileEvent(
                        tick, entities[event.colliders[0]], event.colliders[1], tile_position, event.collision_direction
                    ))

                past_events.add(event)

                collision_remaining = True

        for entity, time in zip(entities, local_times):
            updated_position = entity.bounding_box.position + entity.speed * (1 - time)
            entity.position = updated_position

    def spawn(self, world_object: WorldObject) -> None:
        """
        Spawns a new world object.

        Parameters
        ----------
        world_object: WorldObject
            The world object to spawn.
        """

        self.world_objects.append(world_object)


class LogicLoop:
    """
    The main game loop function.

    This object is callable, hence is should be called as if it was a function. When called, the loop starts and run the
    logic function at each tick, at a specified tick rate.

    Methods
    -------
    stop()
        Stops the main loop.
    """

    def __init__(self, tick_per_second: float, function_logic: callable, function_error: callable):
        """
        Initializes the LogicLoop.

        Parameters
        ----------
        tick_per_second: float
            The tick rate of the loop. It correspond to the number of times the logic will be performed per second.
        function_logic: callable
            The logic function of the loop called each tick.
        function_error: callable
            The error function called whenever an exception occurs within the loop.
        """

        self._function_logic = function_logic
        self._function_error = function_error
        self._tick_period = 1.0 / tick_per_second

        self._tick = 0
        self._running = False

    def __call__(self, number_of_ticks: int = None, max_speed: bool = False) -> None:
        """
        Runs the main logic function.

        Parameters
        ----------
        number_of_ticks: int, optional
            The number of ticks over which the logic will be performed. Il this parameter is not set, the loop will run
            until the stop function is called.
        max_speed: bool, optional
            Ignores the tick rate and runs at maximum speed if set to True.
        """

        target_tick = self._tick + number_of_ticks if number_of_ticks else -1

        current_time = LogicLoop._get_current_time()
        reference_logic = current_time

        self._running = True

        try:
            if max_speed:
                while self._running and self._tick != target_tick:
                    self._do_logic()
            else:
                while self._running and self._tick != target_tick:
                    if LogicLoop._get_current_time() - reference_logic >= self._tick_period:
                        self._do_logic()

                        reference_logic += self._tick_period

                    sleep_time = self._tick_period + reference_logic - LogicLoop._get_current_time()

                    if sleep_time > 0:
                        sleep(sleep_time)
        except BaseException:
            self._function_error()

            raise

        self._running = False

    def _do_logic(self) -> None:
        """
        Calls the logic routine.
        """

        self._function_logic(self._tick)
        self._tick += 1

    def stop(self) -> None:
        """
        Stops the main loop.
        """

        self._running = False

    @staticmethod
    def _get_current_time() -> float:
        """
        Returns the current system time in second.

        Returns
        -------
        time: float
            The current system time.
        """

        return datetime.datetime.utcnow().timestamp()
