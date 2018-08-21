import inspect
import six

__version__ = "0.2.1"


def has_kwargs(function):
    r'''Determine whether a function has \*\*kwargs.

    Parameters
    ----------
    function : callable
        The function to test

    Returns
    -------
    True if function accepts arbitrary keyword arguments.
    False otherwise.
    '''

    if six.PY2:
        return inspect.getargspec(function).keywords is not None
    else:
        sig = inspect.signature(function)

        for param in sig.parameters.values():
            if param.kind == param.VAR_KEYWORD:
                return True

        return False


def filter_kwargs(_function, *args, **kwargs):
    """Given a function and args and keyword args to pass to it, call the function
    but using only the keyword arguments which it accepts.  This is equivalent
    to redefining the function with an additional \*\*kwargs to accept slop
    keyword args.

    If the target function already accepts \*\*kwargs parameters, no filtering
    is performed.

    Parameters
    ----------
    _function : callable
        Function to call.  Can take in any number of args or kwargs

    """

    if has_kwargs(_function):
        return _function(*args, **kwargs)

    # Get the list of function arguments
    func_code = six.get_function_code(_function)
    function_args = func_code.co_varnames[:func_code.co_argcount]
    # Construct a dict of those kwargs which appear in the function
    filtered_kwargs = {}
    for kwarg, value in list(kwargs.items()):
        if kwarg in function_args:
            filtered_kwargs[kwarg] = value
    # Call the function with the supplied args and the filtered kwarg dict
    return _function(*args, **filtered_kwargs)
