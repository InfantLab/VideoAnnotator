import{c as s,p as n,Q as c,y as a}from"./index-DdpWUhBO.js";/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const r=s("Bookmark",[["path",{d:"m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z",key:"1fy3hk"}]]);function o(){const e=n({queryKey:[...c.ingestAccess,a.baseURL],queryFn:()=>a.getIngestAccess(),retry:!1,staleTime:6e4,refetchOnWindowFocus:!1});return{access:e.data??null,isLoading:e.isLoading,sameMachine:e.data?.same_machine??!1,canReadInPlace:e.data?.can_read_in_place??!1,canOpenFolders:e.data?.can_open_folders??!1}}export{r as B,o as u};
