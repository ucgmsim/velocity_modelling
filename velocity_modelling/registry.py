from pathlib import Path

DATA_ROOT = Path(__file__).parent / "Data"
nzvm_registry_path = DATA_ROOT/'nzvm_registry.yaml'

def nzvm_registry_get_vm(version):
    vminfo_list = nzvm_registry.get('vm')
    for vminfo in vminfo_list:
        if str(vminfo['version']) == version:
            return vminfo
    return None

def nzvm_registry_get_basin(basin_name):
    basin_list = nzvm_registry.get('basin')
    for basin in basin_list:
        if basin['name'] == basin_name:
            return basin
    return None

def nzvm_registry_get_surface(surf_name):
    surf_list = nzvm_registry.get('surface')
    for surf in surf_list:
        if surf['name'] == surf_name:
            return surf
    return None

def nzvm_registry_get_tomography(tomo_name):
    tomo_list = nzvm_registry.get('tomography')
    for tomo in tomo_list:
        if tomo['name'] == tomo_name:
            return tomo
    return None

def nzvm_registry_get_vm1d(vm1d_name):
    vm1d_list = nzvm_registry.get('vm1d')
    for vm1d in vm1d_list:
        if vm1d['name'] == vm1d_name:
            return vm1d
    return None

def nzvm_registry_get_submodel(submodel_name):
    submodel_list = nzvm_registry.get('submodel')
    for submodel in submodel_list:
        if submodel['name'] == submodel_name:
            return submodel
    return None