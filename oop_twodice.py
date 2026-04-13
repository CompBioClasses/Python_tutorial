import numpy as np

class TwoDice:
    """Represents a pair of dice, each with NUM_SIDES faces.

    This class demonstrates core object-oriented programming concepts:
      - The constructor (__init__) sets up each instance's initial state
      - Instance attributes (self.x, self.y) store per-object data
      - A name-mangled attribute (self._target_number) signals "internal use"
      - Regular methods (roll, is_success) define object behaviors
      - The __str__ method controls how the object prints
      - Properties (@property) expose computed values as if they were attributes
      - A property setter (@dsum.setter) lets you *assign* to a computed attribute

    Attributes:
        x (int): The face value of the first die (1 to NUM_SIDES).
        y (int): The face value of the second die (1 to NUM_SIDES).
    """

    # Class attribute: belongs to the class itself, not to any single instance.
    # Every TwoDice object shares this value. You can access it as either
    # TwoDice.NUM_SIDES or self.NUM_SIDES (as long as you haven't shadowed it
    # by assigning a different value to self.NUM_SIDES on a specific instance).
    NUM_SIDES = 6

    def __init__(self, d1=None, d2=None):
        """Create a TwoDice instance, optionally with fixed starting values.

        If a die value is not supplied, it is rolled randomly.
        A random target number is also chosen privately at creation time.

        Args:
            d1 (int, optional): Starting face value for the first die (1 to NUM_SIDES).
            d2 (int, optional): Starting face value for the second die (1 to NUM_SIDES).

        Example:
            >>> dice = TwoDice()          # both dice rolled randomly
            >>> fixed = TwoDice(3, 5)     # first die = 3, second die = 5
        """
        # Use the supplied value if given; otherwise roll the die
        if d1 is not None:
            self.x = d1
        else:
            self.x = np.random.randint(1, self.NUM_SIDES + 1)   # randint(a, b) returns a <= n < b

        if d2 is not None:
            self.y = d2
        else:
            self.y = np.random.randint(1, self.NUM_SIDES + 1)

        # The leading underscore is a Python convention meaning "treat this as
        # private" — it signals to other programmers not to access it directly.
        # Python does not enforce true privacy, but the convention is widely respected.
        self._target_number = np.random.randint(1, 2 * self.NUM_SIDES)

    # ------------------------------------------------------------------ #
    # Dunder (double-underscore) methods                                   #
    # ------------------------------------------------------------------ #

    def __str__(self):
        """Return a human-readable string describing the current dice values.

        Python calls this method automatically whenever str() or print() is
        used on an instance of the class. Defining __str__ is how you control
        what your object looks like when displayed.

        Returns:
            str: A sentence reporting both die faces.
        """
        return 'The dice are {} and {}.'.format(self.x, self.y)

    # ------------------------------------------------------------------ #
    # Regular methods                                                       #
    # ------------------------------------------------------------------ #

    def roll(self):
        """Re-roll both dice, replacing their current values with new random ones."""
        self.x = np.random.randint(1, self.NUM_SIDES + 1)
        self.y = np.random.randint(1, self.NUM_SIDES + 1)

    def is_success(self, target):
        """Check whether the sum of the dice meets or exceeds a given target.

        Args:
            target (int): The threshold to compare against the dice sum.

        Returns:
            bool: True if x + y >= target, False otherwise.
        """
        return self.x + self.y >= target

    # ------------------------------------------------------------------ #
    # Properties                                                            #
    # ------------------------------------------------------------------ #

    @property
    def test(self):
        """bool: True if the current dice sum meets the object's internal target.

        This is a *read-only* property — there is no setter, so attempting
        ``dice.test = value`` will raise an AttributeError.

        The @property decorator lets you access this like a plain attribute
        (``dice.test``) even though it runs a method behind the scenes.
        """
        return self.dsum >= self._target_number

    @property
    def dsum(self):
        """int: The sum of both dice faces (read or write via the setter below).

        Using @property here means callers write ``dice.dsum`` rather than
        ``dice.dsum()`` — it looks like a data attribute but behaves like a
        method. This is useful when a value is always derived from other
        attributes and should never be stored separately.
        """
        return self.x + self.y

    @dsum.setter
    def dsum(self, num):
        """Set the dice to a specific sum by choosing x randomly and computing y.

        The setter is paired with the @property above using the decorator
        ``@dsum.setter``. It is invoked whenever you write ``dice.dsum = num``.
        Because two dice must each show 1 to NUM_SIDES, only sums from 2 to
        2*NUM_SIDES are valid.

        The logic picks x from the range of values that keep both dice legal:
          - If num < NUM_SIDES+1,  x can range from 1 to num-1          (so y = num - x >= 1).
          - If num >= NUM_SIDES+1, x can range from num-NUM_SIDES to NUM_SIDES (so y = num - x <= NUM_SIDES).

        Args:
            num (int): The desired total (should be between 2 and 2*NUM_SIDES inclusive).
        """

        if num < 2 or num > 2 * self.NUM_SIDES:
            raise ValueError('Sum must be between 2 and {}'.format(2 * self.NUM_SIDES))

        if num < self.NUM_SIDES + 1:
            self.x = np.random.randint(1, num)                              # x in [1, num-1]
        else:
            self.x = np.random.randint(num - self.NUM_SIDES, self.NUM_SIDES + 1)  # x in [num-NUM_SIDES, NUM_SIDES]
        self.y = num - self.x
