import{c as a,am as t,p as o,O as i,y,Q as u}from"./index-BVM3D6UD.js";/**
 * @license lucide-react v0.462.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */const l=a("HardDrive",[["line",{x1:"22",x2:"2",y1:"12",y2:"12",key:"1y58io"}],["path",{d:"M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z",key:"oot6mr"}],["line",{x1:"6",x2:"6.01",y1:"16",y2:"16",key:"sgf278"}],["line",{x1:"10",x2:"10.01",y1:"16",y2:"16",key:"1l4acy"}]]),c=300*1e3;function p(){const e=t(),r=o({queryKey:u.currentUser,queryFn:()=>y.getCurrentUser(),enabled:e,staleTime:c,retry:!1}),n=r.isError&&r.error instanceof i&&r.error.status===404,s=r.data?r.data.isAdmin:"unknown";return{currentUser:r.data??null,isAdmin:s,isLoading:e&&r.isLoading,endpointUnsupported:n,error:r.error}}export{l as H,p as u};
