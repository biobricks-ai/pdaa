import chembl_downloader as cd

# grab and unpack the latest release (34 as of July 2025)
db_path = cd.download_extract_sqlite()        # ~ 3.8 GB download, ~11 GB unpacked
print(db_path)

# # run a query
# with cd.cursor() as cur:                      # context mgr auto-connects
#     cur.execute("SELECT COUNT(*) FROM activities")
#     print(cur.fetchone())
