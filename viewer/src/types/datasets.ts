/**
 * Saved datasets (VideoAnnotator specs 007 and 018): a named list of videos
 * (names, sizes, paths), never the videos themselves, shared with everyone on
 * the server. Either uploaded from the browser, or a folder the server reads.
 */
export interface ManifestEntry {
  filename: string;
  size_bytes: number;
  /** Path within the dataset's folder; absent in datasets saved before spec 018. */
  relative_path?: string | null;
  last_seen_at?: string | null;
}

export interface SavedDataset {
  id: string;
  name: string;
  description?: string | null;
  owner_user_id: string;
  owner_name?: string | null;
  video_manifest: ManifestEntry[];
  server_folder?: string | null;
  server_folder_recursive?: boolean;
  created_at: string;
  updated_at?: string | null;
  last_used_at?: string | null;
}

export interface DatasetListResponse {
  datasets: SavedDataset[];
  total: number;
}

export interface DatasetCreateRequest {
  name: string;
  description?: string | null;
  video_manifest: ManifestEntry[];
  server_folder?: string | null;
  server_folder_recursive?: boolean;
}

export type DatasetUpdateRequest = Partial<Pick<DatasetCreateRequest, 'name' | 'description' | 'video_manifest'>>;

export interface ScannedVideo {
  relative_path: string;
  name: string;
  size_bytes: number;
}

export interface ServerFolderScan {
  path: string;
  recursive: boolean;
  videos: ScannedVideo[];
}
