# pip install cdsapi rich
import cdsapi
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

c = cdsapi.Client()

# Set your area if you want (N, W, S, E). Example: pan-Europe box.
ECAC_AREA = [75, -20, 25, 60]  # adjust or remove this line to get global

HOURS_UTC = [f"{h:02d}:00" for h in range(24)]
DATES_TO_DOWNLOAD = ["17/07/2023", "18/07/2023", "20/07/2023", "29/07/2023", \
                        "03/08/2023", "05/08/2023"]

def dl_era5_single_levels(dates: list[str], outdir="data/wx"):
    """
    Download ERA5 single-level data for specific dates.
    Creates one file per date.
    
    Args:
        dates: List of dates in format "DD/MM/YYYY" (e.g., ["17/07/2023", "18/07/2023"])
        outdir: Output directory path
    
    Returns:
        List of output file paths (one per date)
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    output_files = []
    
    for date_str in dates:
        day, month, year = date_str.split('/')
        
        # Create filename based on single date
        date_range = f"{year}{month}{day}"
        out = outdir / f"era5_single_hourly_{date_range}.nc"
        
        req = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "year": year,
            "month": month,
            "day": day,
            "time": HOURS_UTC,
            "variable": [
                "convective_available_potential_energy",
                "convective_inhibition",
                "convective_precipitation",
                "instantaneous_10m_wind_gust",
                "total_column_water_vapour",
                "total_cloud_cover",
            ],
        }
        # Optional geographic subset:
        req["area"] = ECAC_AREA
        c.retrieve("reanalysis-era5-single-levels", req, str(out))
        output_files.append(out)
    
    return output_files

def dl_era5_pressure_levels(dates: list[str], outdir="data/wx"):
    """
    Download ERA5 pressure-level data for specific dates.
    Creates one file per date.
    
    Args:
        dates: List of dates in format "DD/MM/YYYY" (e.g., ["17/07/2023", "18/07/2023"])
        outdir: Output directory path
    
    Returns:
        List of output file paths (one per date)
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    output_files = []
    
    for date_str in dates:
        day, month, year = date_str.split('/')
        
        # Create filename based on single date
        date_range = f"{year}{month}{day}"
        out = outdir / f"era5_pl_hourly_{date_range}.nc"
        
        req = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "year": year,
            "month": month,
            "day": day,
            "time": HOURS_UTC,
            "pressure_level": ["1000","925","850","700","600","500"],
            "variable": [
                "u_component_of_wind",
                "v_component_of_wind",
                "vertical_velocity",
            ],
        }
        req["area"] = ECAC_AREA
        c.retrieve("reanalysis-era5-pressure-levels", req, str(out))
        output_files.append(out)
    
    return output_files

if __name__ == "__main__":
    from rich.console import Console
    
    console = Console()
    console.print("[bold green]Starting ERA5 downloads...[/bold green]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Download single-levels
        task1 = progress.add_task(
            "[cyan]Downloading ERA5 single-levels...",
            total=100
        )
        dl_era5_single_levels(DATES_TO_DOWNLOAD)
        progress.update(task1, completed=100)
        
        # Download pressure-levels
        task2 = progress.add_task(
            "[cyan]Downloading ERA5 pressure-levels...",
            total=100
        )
        dl_era5_pressure_levels(DATES_TO_DOWNLOAD)
        progress.update(task2, completed=100)
    
    console.print("\n[bold green]✓ All downloads completed![/bold green]")

# Example: June 2022
# dl_era5_single_levels(2022, 6)
# dl_era5_pressure_levels(2022, 6)
