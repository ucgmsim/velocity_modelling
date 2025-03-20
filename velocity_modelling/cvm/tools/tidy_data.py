import os
import shutil
from pathlib import Path
import re

# Base paths
source_base = "~/Velocity-Model/Data"  # Where the original files are located
dest_base = "~/velocity_modelling/velocity_modelling/cvm/data/basin"  # Where to create new folders
remote_base = "https://github.com/ucgmsim/Velocity-Model/tree/main/Data"

# Surface path mapping
surface_paths = {
    "CantDEM": "DEM/CantDEM.in",
    "PlioceneTop_46_v8p9p18": "Canterbury_Basin/Pre_Quaternary/Pliocene_46_v8p9p18.in",
    "MioceneTop": "Canterbury_Basin/Pre_Quaternary/MioceneTop.in",
    "PaleogeneTop": "Canterbury_Basin/Pre_Quaternary/PaleogeneTop.in",
    "BasementTopSurf": "Canterbury_Basin/Quaternary/BasementTop.in",
    "NZ_DEM": "DEM/NZ_DEM_HD.in",
    "NorthCanterburyBasement": "SI_BASINS/NorthCanterbury_Basement_WGS84_v0p0.in",
    "BPVTop": "Canterbury_Basin/BPV/BPVTop.in",
    "KaikouraBasement": "SI_BASINS/Kaikoura_Basement_WGS84_v0p0.in",
    "CheviotBasement": "SI_BASINS/Cheviot_Basement_WGS84_v0p0.in",
    "HanmerBasement": "SI_BASINS/Hanmer_Basement_WGS84_v0p0.in",
    "MarlboroughBasement": "SI_BASINS/Marlborough_Basement_WGS84_v0p1.in",
    "NelsonBasement": "SI_BASINS/Nelson_Basement_WGS84_v0p0.in",
    "WellingtonBasement_v21p8": "Basins/Wellington/v21p8/Wellington_Grid_WGS84_Hill20210823_100_extracted_d.in",
    "WaikatoHaurakiBasement": "NI_BASINS/WaikatoHaurakiBasin_WGS84_500m_v2019v07v05.in",
    "WanakaBasement": "USER20_BASINS/wanaka_basin_grid_WGS84_v2.in",
    "MacKenzieBasement": "USER20_BASINS/mackenzie_proj_grid_WGS84.in",
    "WakatipuBasement": "USER20_BASINS/WakatipuBasin_WGS84_500m_v2020v07v06.in",
    "RanAlexBasement": "USER20_BASINS/ran-alex_proj_WGS84.in",
    "MosgielBasement": "USER20_BASINS/mos_proj_WGS84.in",
    "BalcluthaBasement": "USER20_BASINS/bal_proj_WGS84.in",
    "DunedinBasement": "USER20_BASINS/dun_proj_WGS84.in",
    "MurchisonBasement": "USER20_BASINS/MurchisonBasin_WGS84_500m_v2020v07v15.in",
    "WaitakiBasement": "USER20_BASINS/wai-hak_WGS84.in",
    "HakatarameaBasement": "USER20_BASINS/wai-hak_WGS84.in",
    "KarameaBasement": "USER20_BASINS/KarameaBasin_WGS84_500m_v12v11v2020.in",
    "CollingwoodBasement": "USER20_BASINS/CollTakBasin_WGS84_500m_v11v11v2020.in",
    "SpringsJunctionBasement": "USER20_BASINS/SpringsJunctionBasin_WGS84_500m_v12v11v2020.in",
    "HawkesBayBasement": "Basins/Napier_Hawkes_Bay/v21p7/HawkesBay_Grid_WGS84_Export.in",
    "NapierBasement": "Basins/Napier_Hawkes_Bay/v21p7/Napier_Grid_WGS84_Export.in",
    "GreaterWellingtonBasement": "Basins/Greater_Wellington_and_Porirua/v21p7/Greater_Wellington_Elevation_WGS84.in",
    "PoriruaBasement": "Basins/Greater_Wellington_and_Porirua/v21p7/Porirua_Elevation_WGS84.in",
    "GisborneBasement": "Basins/Gisborne/v21p11/Gisborne_Surface_Export.in",
    "SouthernHawkesBayBasement": "Basins/Southern_Hawkes_Bay/v21p12/SHB_Surface_Export.in",
    "WairarapaBasement": "Basins/Wairarapa/v21p12/Wairarapa_Surface_Export.in",
    "MotuBayBasement": "Basins/East_Cape/v22p3/Motu_river_Surface_Export.in",
    "WhangaparoaBasement": "USER20_BASINS/Whangaparoa_surface_WGS84.txt",
    "TeAnauBasement": "STUDENTS_BASINS/TeAnau_surface_WGS84.in",
    "WestportBasement": "STUDENTS_BASINS/Westport_surface_WGS84.in"
}

# Basin data (latest versions only)
basins = {
    "Canterbury_Pre_Quaternary_v19p1": {
        "boundaries": ["Boundaries/NewCanterburyBasinBoundary_WGS84_1m.txt"],
        "surfaces": ["CantDEM", "PlioceneTop_46_v8p9p18", "MioceneTop", "PaleogeneTop", "BasementTopSurf"],
        "smoothing": "Boundaries/Smoothing/Canterbury_Pre_Quaternary_v19p1.txt"
    },
    "Canterbury_North_v19p1": {
        "boundaries": ["SI_BASINS/NorthCanterbury_Polygon_WGS84.txt"],
        "surfaces": ["NZ_DEM", "NorthCanterburyBasement"],
        "smoothing": None
    },
    "Banks_Peninsula_Volcanics_v19p1": {
        "boundaries": ["Boundaries/BPVBoundary.txt"],
        "surfaces": ["BPVTop", "MioceneTop"],
        "smoothing": None
    },
    "Kaikoura_v19p1": {
        "boundaries": ["SI_BASINS/Kaikoura_Polygon_WGS84.txt"],
        "surfaces": ["NZ_DEM", "KaikouraBasement"],
        "smoothing": "Boundaries/Smoothing/Kaikoura_v19p1.txt"
    },
    "Cheviot_v19p1": {
        "boundaries": ["SI_BASINS/Cheviot_Polygon_WGS84.txt"],
        "surfaces": ["NZ_DEM", "CheviotBasement"],
        "smoothing": "Boundaries/Smoothing/Cheviot_v19p1.txt"
    },
    "Hanmer_v19p1": {
        "boundaries": ["SI_BASINS/Hanmer_Polygon_WGS84.txt"],
        "surfaces": ["NZ_DEM", "HanmerBasement"],
        "smoothing": None
    },
    "Marlborough_v19p1": {
        "boundaries": ["SI_BASINS/Marlborough_Polygon_WGS84_v0p1.txt"],
        "surfaces": ["NZ_DEM", "MarlboroughBasement"],
        "smoothing": "Boundaries/Smoothing/Marlborough_v19p1.txt"
    },
    "Nelson_v19p1": {
        "boundaries": ["SI_BASINS/Nelson_Polygon_WGS84.txt"],
        "surfaces": ["NZ_DEM", "NelsonBasement"],
        "smoothing": "Boundaries/Smoothing/Nelson_v19p1.txt"
    },
    "Wellington_v21p8": {
        "boundaries": ["Basins/Wellington/v21p8/Wellington_Polygon_Wainuiomata_WGS84.txt"],
        "surfaces": ["NZ_DEM", "WellingtonBasement_v21p8"],
        "smoothing": "Boundaries/Smoothing/Wellington_v21p8.txt"
    },
    "WaikatoHauraki_v19p7": {
        "boundaries": ["Boundaries/WaikatoHaurakiBasinEdge_WGS84.txt"],
        "surfaces": ["NZ_DEM", "WaikatoHaurakiBasement"],
        "smoothing": "Boundaries/Smoothing/WaikatoHauraki_v19p7.txt"
    },
    "Wanaka_v20p6": {
        "boundaries": ["USER20_BASINS/WanakaOutlineWGS84.txt"],
        "surfaces": ["NZ_DEM", "WanakaBasement"],
        "smoothing": None
    },
    "MacKenzie_v20p6": {
        "boundaries": ["USER20_BASINS/mackenzie_basin_outline_nzmg.txt"],
        "surfaces": ["NZ_DEM", "MacKenzieBasement"],
        "smoothing": None
    },
    "Wakatipu_v20p7": {
        "boundaries": ["USER20_BASINS/WakatipuBasinOutlineWGS84.txt"],
        "surfaces": ["NZ_DEM", "WakatipuBasement"],
        "smoothing": None
    },
    "Alexandra_v20p7": {
        "boundaries": ["USER20_BASINS/alexandra_outline.txt"],
        "surfaces": ["NZ_DEM", "RanAlexBasement"],
        "smoothing": None
    },
    "Ranfurly_v20p7": {
        "boundaries": ["USER20_BASINS/ranfurly_outline.txt"],
        "surfaces": ["NZ_DEM", "RanAlexBasement"],
        "smoothing": None
    },
    "NE_Otago_v20p7": {
        "boundaries": [
            "USER20_BASINS/NE_otago/NE_otago_A_outline.txt",
            "USER20_BASINS/NE_otago/NE_otago_B_outline.txt",
            "USER20_BASINS/NE_otago/NE_otago_C_outline.txt",
            "USER20_BASINS/NE_otago/NE_otago_D_outline.txt",
            "USER20_BASINS/NE_otago/NE_otago_E_outline.txt"
        ],
        "surfaces": ["NZ_DEM", "RanAlexBasement"],
        "smoothing": None
    },
    "Mosgiel_v20p7": {
        "boundaries": ["USER20_BASINS/mos_outline_WGS84.txt"],
        "surfaces": ["NZ_DEM", "MosgielBasement"],
        "smoothing": None
    },
    "Balclutha_v20p7": {
        "boundaries": ["USER20_BASINS/bal_outline_WGS84.txt"],
        "surfaces": ["NZ_DEM", "BalcluthaBasement"],
        "smoothing": "Boundaries/Smoothing/Balclutha_v20p7.txt"
    },
    "Dunedin_v20p7": {
        "boundaries": ["USER20_BASINS/dun_outline_WGS84.txt"],
        "surfaces": ["NZ_DEM", "DunedinBasement"],
        "smoothing": "Boundaries/Smoothing/Dunedin_v20p7.txt"
    },
    "Murchison_v20p7": {
        "boundaries": ["USER20_BASINS/Murchison_Basin_Outline_v1_WGS84.txt"],
        "surfaces": ["NZ_DEM", "MurchisonBasement"],
        "smoothing": None
    },
    "Waitaki_v20p8": {
        "boundaries": ["USER20_BASINS/waitaki_outline_WGS84.txt"],
        "surfaces": ["NZ_DEM", "WaitakiBasement"],
        "smoothing": "Boundaries/Smoothing/Waitaki_v20p8.txt"
    },
    "Hakataramea_v20p8": {
        "boundaries": ["USER20_BASINS/hakataramea_outline_WGS84.txt"],
        "surfaces": ["NZ_DEM", "HakatarameaBasement"],
        "smoothing": None
    },
    "Karamea_v20p11": {
        "boundaries": ["USER20_BASINS/Karamea_basin_outline_v1_WGS84.txt"],
        "surfaces": ["NZ_DEM", "KarameaBasement"],
        "smoothing": "Boundaries/Smoothing/Karamea_v20p11.txt"
    },
    "Collingwood_v20p11": {
        "boundaries": [
            "USER20_BASINS/CollingwoodBasinOutline_1_WGS84_v1.txt",
            "USER20_BASINS/CollingwoodBasinOutline_2_WGS84_v1.txt",
            "USER20_BASINS/CollingwoodBasinOutline_3_WGS84_v1.txt"
        ],
        "surfaces": ["NZ_DEM", "CollingwoodBasement"],
        "smoothing": "Boundaries/Smoothing/CollingwoodBasin1_v20p11.txt"
    },
    "SpringsJunction_v20p11": {
        "boundaries": ["USER20_BASINS/SpringsJ_basin_outline_v1_WGS84.txt"],
        "surfaces": ["NZ_DEM", "SpringsJunctionBasement"],
        "smoothing": None
    },
    "HawkesBay_v21p7": {
        "boundaries": [
            "Basins/Napier_Hawkes_Bay/v21p7/HawkesBay1_Outline_WGS84_delim.dat",
            "Basins/Napier_Hawkes_Bay/v21p7/HawkesBay2_Outline_WGS84_delim.dat",
            "Basins/Napier_Hawkes_Bay/v21p7/HawkesBay3_Outline_WGS84_delim.dat",
            "Basins/Napier_Hawkes_Bay/v21p7/HawkesBay4_Outline_WGS84_delim.dat"
        ],
        "surfaces": ["NZ_DEM", "HawkesBayBasement"],
        "smoothing": "Boundaries/Smoothing/HawkesBay1_v21p7.txt"
    },
    "Napier_v21p7": {
        "boundaries": [
            "Basins/Napier_Hawkes_Bay/v21p7/Napier1_Outline_WGS84_delim.dat",
            "Basins/Napier_Hawkes_Bay/v21p7/Napier2_Outline_WGS84_delim.dat",
            "Basins/Napier_Hawkes_Bay/v21p7/Napier3_Outline_WGS84_delim.dat",
            "Basins/Napier_Hawkes_Bay/v21p7/Napier4_Outline_WGS84_delim.dat",
            "Basins/Napier_Hawkes_Bay/v21p7/Napier5_Outline_WGS84_delim.dat",
            "Basins/Napier_Hawkes_Bay/v21p7/Napier6_Outline_WGS84_delim.dat"
        ],
        "sur Surfaces": ["NZ_DEM", "NapierBasement"],
        "smoothing": "Boundaries/Smoothing/Napier1_v21p7.txt"
    },
    "GreaterWellington_v21p7": {
        "boundaries": [
            "Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington1_Outline_WGS84.dat",
            "Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington2_Outline_WGS84.dat",
            "Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington3_Outline_WGS84.dat",
            "Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington4_Outline_WGS84.dat",
            "Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington5_Outline_WGS84.dat",
            "Basins/Greater_Wellington_and_Porirua/v21p7/GreaterWellington6_Outline_WGS84.dat"
        ],
        "surfaces": ["NZ_DEM", "GreaterWellingtonBasement"],
        "smoothing": "Boundaries/Smoothing/GreaterWellington1_v21p7.txt"
    },
    "Porirua_v21p7": {
        "boundaries": [
            "Basins/Greater_Wellington_and_Porirua/v21p7/Porirua1_Outline_WGS84.dat",
            "Basins/Greater_Wellington_and_Porirua/v21p7/Porirua2_Outline_WGS84.dat"
        ],
        "surfaces": ["NZ_DEM", "PoriruaBasement"],
        "smoothing": "Boundaries/Smoothing/Porirua1_v21p7.txt"
    },
    "Gisborne_v21p11": {
        "boundaries": ["Basins/Gisborne/v21p11/Gisborne_Outline_WGS84_delim.dat"],
        "surfaces": ["NZ_DEM", "GisborneBasement"],
        "smoothing": "Boundaries/Smoothing/Gisborne_v21p11.txt"
    },
    "SouthernHawkesBay_v21p12": {
        "boundaries": ["Basins/Southern_Hawkes_Bay/v21p12/SHB_Outline_WGS84_delim.dat"],
        "surfaces": ["NZ_DEM", "SouthernHawkesBayBasement"],
        "smoothing": None
    },
    "Wairarapa_v21p12": {
        "boundaries": ["Basins/Wairarapa/v21p12/Wairarapa_Outline_WGS84_delim.dat"],
        "surfaces": ["NZ_DEM", "WairarapaBasement"],
        "smoothing": "Boundaries/Smoothing/Wairarapa_v21p12.txt"
    },
    "MotuBay_v22p3": {
        "boundaries": [
            "Basins/East_Cape/v22p3/Motu_bay1_Outline_WGS84.txt",
            "Basins/East_Cape/v22p3/Motu_bay2_Outline_WGS84.txt",
            "Basins/East_Cape/v22p3/Motu_bay3_Outline_WGS84.txt"
        ],
        "surfaces": ["NZ_DEM", "MotuBayBasement"],
        "smoothing": "Boundaries/Smoothing/Motu_Bay1_v22p3.txt"
    },
    "Whangaparoa_v23p4": {
        "boundaries": ["USER20_BASINS/Whangaparoa_outline_WGS84.txt"],
        "surfaces": ["NZ_DEM", "WhangaparoaBasement"],
        "smoothing": None
    },
    "TeAnau_v24p9": {
        "boundaries": ["STUDENTS_BASINS/TeAnau_outline_WGS84.txt"],
        "surfaces": ["NZ_DEM", "TeAnauBasement"],
        "smoothing": None
    },
    "Westport_v24p9": {
        "boundaries": ["STUDENTS_BASINS/Westport_outline_WGS84.txt"],
        "surfaces": ["NZ_DEM", "WestportBasement"],
        "smoothing": None
    }
}

def remove_version(basin_name):
    return re.sub(r'_v\d+p\d+$', '', basin_name)

# Process each basin
for basin_name, data in basins.items():
    try:
        # Remove version from folder name
        folder_name = remove_version(basin_name)
        folder_path = os.path.expanduser(os.path.join(dest_base, folder_name))
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")

        # Prepare README content
        readme_content = f"{basin_name}\n"
        readme_content += ("="*len(basin_name))+"\n\n"
        readme_content += f"Data have been retrieved from the following remote sources\n\n"
        readme_content += "Boundary Files:\n"
        for boundary in data["boundaries"]:
            remote_path = f"{remote_base}/{boundary}"
            readme_content += f"  {boundary}: {remote_path}\n"
            # Copy boundary file
            src_path = os.path.expanduser(os.path.join(source_base, boundary))
            dest_path = os.path.join(folder_path, os.path.basename(boundary))
            shutil.copy2(src_path, dest_path)
            print(f"Copied boundary: {src_path} -> {dest_path}")

        readme_content += "\nSurface Files:\n"
        for surface in data["surfaces"]:

            surface_path = surface_paths.get(surface, "Unknown path")
            remote_path = f"{remote_base}/{surface_path}"
            readme_content += f"  {surface}: {remote_path}\n"
            if 'DEM' in surface: # skip file copy
                continue
            # Copy surface file
            src_path = os.path.expanduser(os.path.join(source_base, surface_path))
            dest_path = os.path.join(folder_path, os.path.basename(surface_path))
            shutil.copy2(src_path, dest_path)
            print(f"Copied surface: {src_path} -> {dest_path}")

        readme_content += "\nSmoothing File:\n"
        if data["smoothing"]:
            remote_path = f"{remote_base}/{data['smoothing']}"
            smoothing_filename = os.path.basename(data["smoothing"]).replace(".txt", "_smoothing.txt")
            readme_content += f"  {smoothing_filename}: {remote_path}\n"
            # Copy smoothing file with suffix
            src_path = os.path.expanduser(os.path.join(source_base, data["smoothing"]))
            dest_path = os.path.join(folder_path, smoothing_filename)
            shutil.copy2(src_path, dest_path)
            print(f"Copied smoothing: {src_path} -> {dest_path}")
        else:
            readme_content += "  None\n"

        # Write README file
        readme_path = os.path.join(folder_path, "README.txt")
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"Created README: {readme_path}")

    except FileNotFoundError as e:
        print(f"Error: File not found for {basin_name}: {str(e)}")
    except Exception as e:
        print(f"Error processing {basin_name}: {str(e)}")

print("Process completed!")