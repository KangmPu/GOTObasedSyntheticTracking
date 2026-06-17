# Pipeline Workflow

This project runs a daily asteroid candidate pipeline. The daily workflow finds candidate observations, groups them by target, site, and time, generates server-side thumbnails, builds local GIFs, and runs moving-object detection. Detected candidates are passed to the AP workflow, which recovers sky positions from detector tracks, classifies each source, and then routes trail sources to pill forced photometry or point sources to GOTO point-source photometry. The final AP output is a PSV result file.

```mermaid
flowchart TD
    A["Daily Workflow"] --> B["Find candidate observations"]
    B --> C["Group candidates by target, site, and time"]
    C --> D["Generate thumbnails on server"]
    D --> E["Build local GIFs"]
    E --> F["Run moving-object detection"]
    F --> G{"Detection found?"}

    G -- "No" --> H["Store daily review outputs"]
    G -- "Yes" --> I["AP Workflow"]

    I --> J["Recover sky positions from detector tracks"]
    J --> K["Classify source"]

    K -- "Trail source" --> L["Use supplied position and trail geometry"]
    L --> M["Generate FITS cutouts on server"]
    M --> N["Run pill forced photometry"]
    N --> O["Write PSV result"]

    K -- "Point source" --> P["Use supplied position"]
    P --> Q["Query GOTO point-source photometry"]
    Q --> O
```
