import os

# When we call `scipy.special.expit`, return the same array type as the input
# See https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html for more details
os.environ["SCIPY_ARRAY_API"] = "1"
