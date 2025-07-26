import jax
from jax.sharding import PartitionSpec, Mesh
import numpy as np
import jax.numpy as jnp


def mesh_along_dim(
        arr,
        shared_along_dim,
        num_devices
):
    mesh_shape = (num_devices, 1)
    axis_names = []
    for i in range(arr.ndim):
        if i == shared_along_dim:
            axis_names.append(str(i))
        else:
            axis_names.append(None)
    return mesh_shape, axis_names


def multiple_device_sharding(
        arr,
        num_devices: int ,
        mesh_shape: tuple = None,
        shared_along_dim: int = None,
        device: str = "cpu",
        visual: bool = False
        ):
    if mesh_shape is None:
        if not shared_along_dim:
            raise ValueError("since mesh is none so the sharded_along_dim not \
                be None and must be any of the axis of arr")
        else:
            mesh_shape, axis_names = mesh_along_dim(
                arr=arr,
                shared_along_dim=shared_along_dim,
                num_devices=num_devices
            )
    else:
        if shared_along_dim is None:
            import warnings
            warnings.warn("\nThe shared_along_dim is not provided, \
                copying the whole data to each device")
            axis_names = []
            for i in range(arr.ndim):
                    axis_names.append(None) # Is this correct?
        else:
            axis_names = []
            for i in range(arr.ndim):
                if i == shared_along_dim:
                    axis_names.append(str(i))
                else:
                    axis_names.append(None)

    # does not sounds good to me as it makes the mesh 3d
    # mesh_axis_names = tuple(name for name in axis_names if name is not None)
    mesh_axis_names = tuple(str(i) for i in range(arr.ndim))

    assert mesh_shape[0] * mesh_shape[1] == num_devices, f"num_devices = {num_devices} \
        are not equal to the mesh_shape = {mesh_shape}"

    # getting all devices # in our case they will 1 gpu
    devices = jax.devices(device)
    print("DEVICES \n", devices, "\n")

    assert len(devices) == num_devices, f"num_devices = {num_devices} \
        are not equal to the devices = {len(devices)}"

    # aranging devices for mesh
    devices_np = np.array(devices).reshape(*mesh_shape)

    mesh = Mesh(devices_np, mesh_axis_names)
    sharding = jax.sharding.NamedSharding(mesh, PartitionSpec(*axis_names))

    sharded_arr = jax.device_put(arr, sharding)

    if visual == True:
        jax.debug.visualize_array_sharding(sharded_arr)

    return sharded_arr


@jax.jit
def parallel_fn(
    sharded_arr
):
    results = 2 * jnp.sin(sharded_arr) + 1
    return results


@jax.jit
def f_contract(
    sharded_arr
):
    return sharded_arr.sum(axis=0)


def main():
    num_devices = 8

    # this has to be just before any use of jax - we are using jnp for creating array
    jax.config.update('jax_num_cpu_devices', num_devices)

    # making an array
    arr = jnp.arange(32.0).reshape(4, 8)

    sharded_arr = multiple_device_sharding(arr, num_devices=8, \
                    mesh_shape=(2, 4), device="cpu", visual=True)

    results = parallel_fn(sharded_arr)

    assert results.sharding == sharded_arr.sharding

    sharded_sum = f_contract(sharded_arr)

    print("\n")
    jax.debug.visualize_array_sharding(sharded_sum)


if __name__ == "__main__":
    main()