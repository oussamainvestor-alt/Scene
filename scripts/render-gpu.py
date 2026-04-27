#!/usr/bin/env python3
"""
GPU-accelerated headless renderer — NVIDIA L40S via ModernGL EGL.
No browser. No Puppeteer. Directly uses the GPU.
"""

import os, sys, math, json, argparse, subprocess, time
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

PROJECT  = Path(__file__).resolve().parent.parent
TEXTURES = PROJECT / 'public' / 'textures'


# ─────────────────────────── CLI ────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--camera',   required=True)
    p.add_argument('--layout',   required=True)
    p.add_argument('--video',    required=True)
    p.add_argument('--audio',    default=None)
    p.add_argument('--output',   default=str(PROJECT / 'Output' / 'output.mp4'))
    p.add_argument('--width',    type=int,   default=1920)
    p.add_argument('--height',   type=int,   default=1080)
    p.add_argument('--fps',      type=int,   default=30)
    p.add_argument('--bitrate',  default='20M')
    p.add_argument('--preset',   default='p4')
    p.add_argument('--duration', type=float, default=None)
    return p.parse_args()


# ─────────────────────────── MATRIX ─────────────────────────

def perspective(fov_deg, aspect, near=0.1, far=100.0):
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    d = near - far
    return np.array([
        [f/aspect, 0, 0,           0         ],
        [0,        f, 0,           0         ],
        [0,        0, (near+far)/d, 2*far*near/d],
        [0,        0, -1,          0         ],
    ], 'f4')

def look_at(eye, center, up=(0,1,0)):
    e, c, u = (np.asarray(x,'f4') for x in (eye,center,up))
    f = c-e; f /= np.linalg.norm(f)
    r = np.cross(f,u); r /= np.linalg.norm(r)
    u = np.cross(r,f)
    m = np.eye(4, dtype='f4')
    m[0,:3]=r;  m[0,3]=-np.dot(r,e)
    m[1,:3]=u;  m[1,3]=-np.dot(u,e)
    m[2,:3]=-f; m[2,3]= np.dot(f,e)
    return m

def translate(x,y,z):
    m=np.eye(4,dtype='f4'); m[0,3]=x; m[1,3]=y; m[2,3]=z; return m

def scale(s):
    m=np.eye(4,dtype='f4'); m[0,0]=m[1,1]=m[2,2]=s; return m

def rot_x(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]],'f4')

def rot_y(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]],'f4')

def rot_z(a):
    c,s=math.cos(a),math.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],'f4')

def euler_xyz(rx,ry,rz):
    return rot_x(rx) @ rot_y(ry) @ rot_z(rz)


# ─────────────────────────── GEOMETRY ───────────────────────

STRIDE = 8  # floats: pos(3)+norm(3)+uv(2)

def make_sphere(radius, wS, hS):
    verts = []
    for iy in range(hS+1):
        v     = iy/hS
        theta = v*math.pi
        for ix in range(wS+1):
            u   = ix/wS
            phi = u*2*math.pi
            x   = -radius*math.cos(phi)*math.sin(theta)
            y   =  radius*math.cos(theta)
            z   =  radius*math.sin(phi)*math.sin(theta)
            verts += [x,y,z, x/radius,y/radius,z/radius, u,v]
    idx=[]
    for iy in range(hS):
        for ix in range(wS):
            a=ix+(wS+1)*iy; b=ix+(wS+1)*(iy+1)
            c=(ix+1)+(wS+1)*(iy+1); d=(ix+1)+(wS+1)*iy
            if iy!=0:    idx+=[a,b,d]
            if iy!=hS-1: idx+=[b,c,d]
    return np.array(verts,'f4'), np.array(idx,'i4')

def make_plane_xy(w, h, wS, hS):
    verts=[]
    for iy in range(hS+1):
        for ix in range(wS+1):
            u = ix/wS
            v = 1.0-iy/hS
            x = u*w - w*0.5
            y = h*0.5 - iy/hS*h
            verts += [x,y,0, 0,0,1, u,v]
    idx=[]
    for iy in range(hS):
        for ix in range(wS):
            a=ix+(wS+1)*iy; b=ix+(wS+1)*(iy+1)
            c=(ix+1)+(wS+1)*(iy+1); d=(ix+1)+(wS+1)*iy
            idx += [a,b,d, b,c,d]
    return np.array(verts,'f4'), np.array(idx,'i4')

def make_plane_xz(size, segs):
    verts=[]
    for iz in range(segs+1):
        for ix in range(segs+1):
            u = ix/segs
            v = 1.0-iz/segs
            x = u*size-size*0.5
            z = iz/segs*size-size*0.5
            verts += [x,0,z, 0,1,0, u,v]
    idx=[]
    for iz in range(segs):
        for ix in range(segs):
            a=ix+(segs+1)*iz; b=ix+(segs+1)*(iz+1)
            c=(ix+1)+(segs+1)*(iz+1); d=(ix+1)+(segs+1)*iz
            idx += [a,b,d, b,c,d]
    return np.array(verts,'f4'), np.array(idx,'i4')

def make_quad(w, h):
    hw,hh = w*0.5, h*0.5
    v = np.array([
        [-hw,-hh,0, 0,0,1, 0,0],
        [ hw,-hh,0, 0,0,1, 1,0],
        [ hw, hh,0, 0,0,1, 1,1],
        [-hw, hh,0, 0,0,1, 0,1],
    ],'f4')
    return v, np.array([0,1,2,0,2,3],'i4')


# ─────────────────────────── SHADERS ────────────────────────

# All vertex shaders use the same 3-attribute layout so VAO creation is uniform.
# Programs that don't need in_norm/in_uv simply ignore them.

_COMMON_VERT_PREFIX = """
#version 330 core
in vec3 in_vert;
in vec3 in_norm;
in vec2 in_uv;
uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;
"""

BG_VERT = _COMMON_VERT_PREFIX + """
void main() {
    // in_norm referenced so compiler keeps the attribute active
    gl_Position = u_proj * u_view * u_model * vec4(in_vert + in_norm*0.0 + vec3(in_uv,0.0)*0.0, 1.0);
}
"""
BG_FRAG = """
#version 330 core
uniform vec3 u_color;
out vec4 frag_color;
void main(){ frag_color=vec4(u_color,1.0); }
"""

GROUND_VERT = _COMMON_VERT_PREFIX + """
out vec3 vWP;
out vec3 vWN;
out vec2 vUV;
void main(){
    vec4 wp = u_model*vec4(in_vert,1.0);
    vWP = wp.xyz;
    vWN = normalize(mat3(u_model)*in_norm);
    vUV = in_uv;
    gl_Position = u_proj*u_view*wp;
}
"""
GROUND_FRAG = """
#version 330 core
in vec3 vWP; in vec3 vWN; in vec2 vUV;
uniform sampler2D u_albedo;
uniform sampler2D u_normal_map;
uniform sampler2D u_roughness;
uniform float u_tex_repeat;
uniform vec3 u_camera_pos;
uniform vec3 u_ambient;
uniform vec3 u_hemi_sky;
uniform vec3 u_hemi_ground;
uniform vec3 u_dir_dir;
uniform vec3 u_dir_color;
uniform vec3 u_spot_pos;
uniform vec3 u_spot_color;
uniform float u_spot_angle;
uniform float u_spot_penumbra;
uniform float u_spot_distance;
uniform vec3 u_pt1_pos;
uniform vec3 u_pt1_color;
uniform vec3 u_orb_pos;
uniform vec3 u_orb_color;
uniform float u_orb_intensity;
uniform vec3 u_fog_color;
uniform float u_fog_near;
uniform float u_fog_far;
out vec4 frag_color;

float spec_term(vec3 N,vec3 L,vec3 V,float rough){
    vec3 H=normalize(L+V);
    float NdotH=max(dot(N,H),0.0);
    float sh=max(2.0,(1.0-rough)*128.0);
    return pow(NdotH,sh)*(1.0-rough)*0.3;
}

void main(){
    vec2 uv = vUV*u_tex_repeat;
    vec3 alb= pow(texture(u_albedo,uv).rgb,vec3(2.2));
    float rgh= texture(u_roughness,uv).r;
    // Normal mapping — ground TBN: T=(1,0,0), B=(0,0,-1), N=(0,1,0)
    vec3 nt = texture(u_normal_map,uv).rgb*2.0-1.0;
    mat3 TBN=mat3(vec3(1,0,0),vec3(0,0,-1),vec3(0,1,0));
    vec3 N=normalize(TBN*nt);
    vec3 V=normalize(u_camera_pos-vWP);

    float hw=dot(N,vec3(0,1,0))*0.5+0.5;
    vec3 col=alb*(u_ambient+mix(u_hemi_ground,u_hemi_sky,hw));

    // Directional
    vec3 Ld=normalize(u_dir_dir);
    float NLd=max(dot(N,Ld),0.0);
    if(NLd>0.0){ col+=alb*u_dir_color*NLd + u_dir_color*spec_term(N,Ld,V,rgh); }

    // Spot (pointing slightly forward/down)
    vec3 spotDir=normalize(vec3(0.0,-1.0,-0.1));
    vec3 toFrag=vWP-u_spot_pos;
    float ds=length(toFrag);
    float cosA=dot(normalize(toFrag),spotDir);
    float cosOuter=cos(u_spot_angle);
    float cosInner=cos(u_spot_angle*(1.0-u_spot_penumbra));
    if(ds<u_spot_distance && cosA>cosOuter){
        vec3 Ls=normalize(-toFrag);
        float NLs=max(dot(N,Ls),0.0);
        float cone=smoothstep(cosOuter,cosInner,cosA);
        float da=max(0.0,1.0-ds/u_spot_distance); da*=da;
        float at=cone*da;
        col+=alb*u_spot_color*NLs*at + u_spot_color*spec_term(N,Ls,V,rgh)*at;
    }

    // Fixed point light
    vec3 Lp1=u_pt1_pos-vWP; float d1=length(Lp1);
    if(d1<8.0){ float a=max(0.0,1.0-d1/8.0)*max(0.0,1.0-d1/8.0);
        col+=alb*u_pt1_color*max(dot(N,normalize(Lp1)),0.0)*a; }

    // Orb point light (warm orange glow)
    vec3 Lo=u_orb_pos-vWP; float dorb=length(Lo);
    if(dorb<5.0){ float a=max(0.0,1.0-(dorb/5.0)*(dorb/5.0));
        col+=alb*u_orb_color*(u_orb_intensity*0.05)*max(dot(N,normalize(Lo)),0.0)*a; }

    col=pow(max(col,0.0),vec3(1.0/2.2));

    // Fog
    float fd=length(vWP-u_camera_pos);
    float ff=clamp((u_fog_far-fd)/(u_fog_far-u_fog_near),0.0,1.0);
    col=mix(u_fog_color,col,ff);
    frag_color=vec4(col,1.0);
}
"""

ORB_VERT = _COMMON_VERT_PREFIX + """
out vec3 vWP; out vec3 vWN;
void main(){
    vec4 wp=u_model*vec4(in_vert,1.0);
    vWP=wp.xyz; vWN=normalize(mat3(u_model)*in_norm);
    gl_Position=u_proj*u_view*wp;
}
"""
ORB_FRAG = """
#version 330 core
uniform float uEnergy; uniform float uEmitStr; uniform vec3 u_camera_pos;
in vec3 vWP; in vec3 vWN;
out vec4 frag_color;
void main(){
    vec3 N=normalize(vWN);
    vec3 V=normalize(u_camera_pos-vWP);
    float NV=clamp(dot(N,V),0.0,1.0);
    float limb=1.0-NV;
    float iL=pow(NV,0.50);
    vec3 hotspot=vec3(1.00,0.97,0.92);
    vec3 cream  =vec3(1.00,0.88,0.70);
    vec3 peach  =vec3(0.94,0.74,0.50);
    vec3 darkRim=vec3(0.22,0.16,0.10);
    vec3 col=mix(peach,cream,iL);
    col=mix(col,hotspot,iL*iL*0.55);
    col=mix(col,darkRim,pow(limb,1.2)*0.42);
    col+=vec3(0.16,0.08,0.02)*uEnergy*NV;
    col*=uEmitStr;
    float alpha=0.72+pow(limb,2.0)*0.18;
    frag_color=vec4(clamp(col,0.0,1.0),alpha);
}
"""

SCREEN_VERT = _COMMON_VERT_PREFIX + """
uniform float uCurveTop,uCurveBottom,uCurveLeft,uCurveRight;
out vec2 vUv; out vec3 vLocalPos;
void main(){
    vUv=in_uv; vLocalPos=in_vert + in_norm*0.0;  // in_norm ref keeps attribute active
    vec3 wp=in_vert;
    float nx=in_uv.x*2.0-1.0, ny=in_uv.y*2.0-1.0;
    float tW=smoothstep(0.0,1.0,in_uv.y)       *(1.0-nx*nx);
    float bW=smoothstep(0.0,1.0,1.0-in_uv.y)   *(1.0-nx*nx);
    float lW=smoothstep(0.0,1.0,1.0-in_uv.x)   *(1.0-ny*ny);
    float rW=smoothstep(0.0,1.0,in_uv.x)        *(1.0-ny*ny);
    wp.y+=((uCurveTop*tW)-(uCurveBottom*bW))*1.8;
    wp.x+=((uCurveRight*rW)-(uCurveLeft*lW))*1.8;
    float dC=(uCurveTop*tW)+(uCurveBottom*bW)+(uCurveLeft*lW)+(uCurveRight*rW);
    wp.z+=dC*0.9;
    gl_Position=u_proj*u_view*u_model*vec4(wp,1.0);
}
"""
SCREEN_FRAG = """
#version 330 core
in vec2 vUv; in vec3 vLocalPos;
uniform sampler2D uVideoTex;
uniform float uVideoAspect,uScreenAspect,uHasVideo,uRadius,uContentScale;
uniform vec2 uSize; uniform vec3 uBackground;
out vec4 frag_color;
float rRectSdf(vec2 p,vec2 hs,float r){
    vec2 q=abs(p)-(hs-vec2(r));
    return length(max(q,0.0))+min(max(q.x,q.y),0.0)-r;
}
void main(){
    vec2 fu=vec2(vLocalPos.x/uSize.x+0.5, vLocalPos.y/uSize.y+0.5);
    float d=rRectSdf((fu-0.5)*uSize, 0.5*uSize, uRadius);
    if(d>0.0) discard;
    vec3 col=uBackground;
    if(uHasVideo>0.5){
        float vw,vh;
        if(uVideoAspect>uScreenAspect){vw=1.0; vh=uScreenAspect/uVideoAspect;}
        else{vw=uVideoAspect/uScreenAspect; vh=1.0;}
        vw*=uContentScale; vh*=uContentScale;
        float x0=0.5-vw*0.5,y0=0.5-vh*0.5,x1=0.5+vw*0.5,y1=0.5+vh*0.5;
        if(fu.x>=x0&&fu.x<=x1&&fu.y>=y0&&fu.y<=y1){
            vec2 tv=vec2((fu.x-x0)/vw, 1.0-(fu.y-y0)/vh);
            col=texture(uVideoTex,tv).rgb;
        }
    }
    frag_color=vec4(col,1.0);
}
"""


# ─────────────────────────── ORB ANIMATION ──────────────────

class OrbState:
    def __init__(self):
        self.se=0.0; self.sx=0.0; self.sy=0.65; self.sz=0.0; self.ry=0.0

    def update(self, t, nrg, dt):
        if nrg>self.se: self.se+=(nrg-self.se)*min(1.0,dt*20)
        else:           self.se+=(nrg-self.se)*min(1.0,dt*4)
        e=self.se

        bs=1.0+math.sin(t*1.8)*0.015; aus=1.0+e*0.35
        base=0.9; ts=bs*aus*base

        idX=math.sin(t*0.3)*0.1+math.sin(t*0.5)*0.05
        idZ=math.cos(t*0.25)*0.08+math.cos(t*0.4)*0.04
        idY=math.sin(t*0.2)*0.06
        ex=e*1.8
        exX=math.sin(t*2.5)*0.2*ex; exZ=math.cos(t*2.0)*0.15*ex
        exY=abs(math.sin(t*3.5))*0.12*ex
        dr=0.1+e*0.2
        drX=math.sin(t*0.15)*dr; drZ=math.cos(t*0.12)*dr

        tX=idX+exX*0.6+drX; tY=0.65+idY+exY; tZ=idZ+exZ*0.6+drZ
        sp=2.0+e*3.0
        self.sx+=(tX-self.sx)*dt*sp
        self.sy+=(tY-self.sy)*dt*sp
        self.sz+=(tZ-self.sz)*dt*sp
        vx=(tX-self.sx)*sp; vz=(tZ-self.sz)*sp

        lean=1.0+e*0.5
        rz=-vx*lean*0.3-math.sin(t*2.5)*e*0.3
        rx= vz*lean*0.25-math.cos(t*2.0)*e*0.25
        self.ry+=(t*0.015+e*0.1-self.ry)*dt*3.0

        ox=0.0+self.sx*base; oy=0.05+self.sy*base; oz=0.0+self.sz*base
        emit=(1.10+math.sin(t*1.8)*0.08)*(1.0+e*0.7)
        return dict(lp=(ox,oy,oz),rx=rx,ry=self.ry,rz=rz,
                    scale=ts,energy=e,emit=emit,li=6.0+e*18.0)


# ─────────────────────────── HELPERS ────────────────────────

def hex_rgb(h, mul=1.0):
    h=h.lstrip('#')
    return tuple(int(h[i:i+2],16)/255.0*mul for i in(0,2,4))

def load_tex(ctx, path, mip=True):
    img=Image.open(path).convert('RGBA').transpose(Image.FLIP_TOP_BOTTOM)
    t=ctx.texture(img.size, 4, img.tobytes())
    t.repeat_x=t.repeat_y=True
    t.filter=(moderngl.LINEAR_MIPMAP_LINEAR,moderngl.LINEAR) if mip else (moderngl.LINEAR,moderngl.LINEAR)
    if mip: t.build_mipmaps()
    return t

def make_vao(ctx, prog, verts, indices):
    """Build VAO, skipping attributes not present in the compiled program."""
    vbo = ctx.buffer(verts.tobytes())
    ibo = ctx.buffer(indices.tobytes())
    pk  = set(prog)
    # Vertex layout: pos(12 B) | norm(12 B) | uv(8 B)
    fmt, names = [], []
    if 'in_vert' in pk: fmt.append('3f'); names.append('in_vert')
    else:               fmt.append('12x')
    if 'in_norm' in pk: fmt.append('3f'); names.append('in_norm')
    else:               fmt.append('12x')
    if 'in_uv'   in pk: fmt.append('2f'); names.append('in_uv')
    else:               fmt.append('8x')
    return ctx.vertex_array(prog, [(vbo, ' '.join(fmt), *names)], ibo)

def w(prog, name, val):
    if name in prog: prog[name].write(np.array(val,'f4').tobytes())

def s1(prog, name, val):
    if name in prog: prog[name].value=float(val)

def s3(prog, name, val):
    if name in prog: prog[name].value=tuple(val)

def mat_bytes(m):
    return m.T.astype('f4').tobytes()


# ─────────────────────────── AUDIO ──────────────────────────

def audio_energy(src, fps, n_frames):
    print('  Analyzing audio energy...')
    import tempfile
    tmp=tempfile.mktemp(suffix='.pcm')
    try:
        subprocess.run(['ffmpeg','-y','-i',str(src),
            '-f','f32le','-acodec','pcm_f32le','-ar','44100','-ac','1',tmp],
            check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        raw=open(tmp,'rb').read(); os.unlink(tmp)
    except Exception as exc:
        print(f'  Audio failed ({exc}), using default energy')
        return [0.3]*n_frames
    samples=np.frombuffer(raw,'f4')
    spf=int(44100/fps)
    en=np.zeros(n_frames,'f4')
    for fi in range(n_frames):
        ch=samples[fi*spf:fi*spf+spf]
        if len(ch): en[fi]=float(np.sqrt(np.mean(ch*ch)))
    mx=en.max()
    if mx>0: en/=mx
    print('  Audio done.')
    return en.tolist()


# ─────────────────────────── MAIN ───────────────────────────

def main():
    args=parse_args()
    W,H,FPS=args.width,args.height,args.fps
    OUT=Path(args.output); OUT.parent.mkdir(parents=True,exist_ok=True)

    cam=json.loads(Path(args.camera).read_text())
    lay=json.loads(Path(args.layout).read_text())
    vid=Path(args.video)
    aud=Path(args.audio) if args.audio else vid

    def get_dur(p):
        r=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
            '-of','default=noprint_wrappers=1:nokey=1',str(p)],
            capture_output=True,text=True)
        try: return float(r.stdout.strip())
        except: return None

    dur=args.duration or get_dur(vid) or 10.0
    NF=math.ceil(dur*FPS)

    print(f'''
  GPU Renderer (NVIDIA L40S · EGL)
  ══════════════════════════════════
  {W}x{H} @ {FPS}fps  ·  {dur:.1f}s  →  {NF} frames
  Video : {vid.name}
  Audio : {aud.name}
  Out   : {OUT}
''')

    energies=audio_energy(aud, FPS, NF)

    # ── ModernGL EGL ──────────────────────────────────────────
    print('  Creating NVIDIA EGL context...')
    ctx=moderngl.create_standalone_context(backend='egl')
    print(f"  {ctx.info['GL_RENDERER']}  ·  {ctx.info['GL_VERSION']}")

    # FBO
    fbo=ctx.framebuffer([ctx.renderbuffer((W,H))], ctx.depth_renderbuffer((W,H)))
    fbo.use()

    # Programs
    bg_prog  =ctx.program(vertex_shader=BG_VERT,     fragment_shader=BG_FRAG)
    gnd_prog =ctx.program(vertex_shader=GROUND_VERT, fragment_shader=GROUND_FRAG)
    orb_prog =ctx.program(vertex_shader=ORB_VERT,    fragment_shader=ORB_FRAG)
    scr_prog =ctx.program(vertex_shader=SCREEN_VERT, fragment_shader=SCREEN_FRAG)

    # ── Geometry ──────────────────────────────────────────────
    print('  Building geometry...')
    ws=lay.get('worldSize',1)
    bg_vao=make_vao(ctx,bg_prog,  *make_quad(22*ws,12*ws))
    gnd_vao=make_vao(ctx,gnd_prog,*make_plane_xz(ws*10, max(32,ws*16)))
    orb_vao=make_vao(ctx,orb_prog,*make_sphere(0.44,64,48))

    sar=lay['screen']['aspectRatio']
    sa=max(1,sar[0])/max(1,sar[1])
    sw=2.7; sh=sw/sa
    scr_vao=make_vao(ctx,scr_prog,*make_plane_xy(sw,sh,64,64))

    # ── Textures ──────────────────────────────────────────────
    print('  Loading ground textures...')
    t_alb =load_tex(ctx, TEXTURES/'asphalt_basecolor.png')
    t_nrm =load_tex(ctx, TEXTURES/'asphalt_normal.png',    mip=False)
    t_rgh =load_tex(ctx, TEXTURES/'asphalt_roughness.png', mip=False)
    t_alb.use(0); t_nrm.use(1); t_rgh.use(2)

    # Video texture (unit 3), sized for source video (1080×1080)
    VW,VH=1080,1080
    vid_tex=ctx.texture((VW,VH),3)
    vid_tex.repeat_x=vid_tex.repeat_y=False
    vid_tex.filter=moderngl.LINEAR,moderngl.LINEAR
    vid_tex.use(3)

    # ── Projection / View (constant) ─────────────────────────
    proj=perspective(40.0, W/H)
    view=look_at(cam['position'], cam['target'])
    P=mat_bytes(proj); V=mat_bytes(view)
    cam_pos=tuple(cam['position'])

    # ── Scene layout ─────────────────────────────────────────
    gr   =lay.get('groupRotation',0.0)
    Mg   =rot_y(gr)                          # outer group rotation
    Mscr =(Mg @ translate(*lay['screen']['position'])
              @ scale(lay['screen']['scale']))
    tex_rep=ws*10/2.5

    edge =lay['screen'].get('edgeCurve',{})
    cs   =lay['screen'].get('contentScale',1.0)
    br   =min(max(0,lay['screen'].get('borderRadius',0.08)),min(sw,sh)*0.45)
    vid_ar=VW/VH

    # ── Constant lighting for ground ─────────────────────────
    bg_z=-9*ws; bg_y=3.8*ws
    Mbg=translate(0,bg_y,bg_z)

    gnd_prog['u_albedo'].value    =0
    gnd_prog['u_normal_map'].value=1
    gnd_prog['u_roughness'].value =2
    s1(gnd_prog,'u_tex_repeat',tex_rep)
    s3(gnd_prog,'u_camera_pos',cam_pos)
    s3(gnd_prog,'u_ambient',   hex_rgb('#162133',0.08))
    s3(gnd_prog,'u_hemi_sky',  hex_rgb('#132238',0.18))
    s3(gnd_prog,'u_hemi_ground',hex_rgb('#090d16',0.18))
    dir_p=np.array([2.5,4.2,1.8],'f4'); dir_d=dir_p/np.linalg.norm(dir_p)
    s3(gnd_prog,'u_dir_dir',   tuple(dir_d))
    s3(gnd_prog,'u_dir_color', hex_rgb('#7a98c0',0.6))
    s3(gnd_prog,'u_spot_pos',  (0.0,5.8,0.6))
    s3(gnd_prog,'u_spot_color',hex_rgb('#b5c9ef',7.0))
    s1(gnd_prog,'u_spot_angle',0.58)
    s1(gnd_prog,'u_spot_penumbra',0.9)
    s1(gnd_prog,'u_spot_distance',13.0)
    s3(gnd_prog,'u_pt1_pos',   (-2.5,1.4,-2.0))
    s3(gnd_prog,'u_pt1_color', hex_rgb('#41618f',0.7))
    s3(gnd_prog,'u_orb_color', hex_rgb('#ffaa44'))
    s3(gnd_prog,'u_fog_color', hex_rgb('#060a11'))
    s1(gnd_prog,'u_fog_near',  5.0)
    s1(gnd_prog,'u_fog_far',   18.0)

    scr_prog['uVideoTex'].value    =3
    s1(scr_prog,'uVideoAspect',  vid_ar)
    s1(scr_prog,'uScreenAspect', sa)
    s1(scr_prog,'uHasVideo',     1.0)
    s1(scr_prog,'uRadius',       br)
    scr_prog['uSize'].value        =(sw,sh)
    s3(scr_prog,'uBackground',   (0.03,0.04,0.07))
    s1(scr_prog,'uCurveTop',     float(edge.get('top',0)))
    s1(scr_prog,'uCurveBottom',  float(edge.get('bottom',0)))
    s1(scr_prog,'uCurveLeft',    float(edge.get('left',0)))
    s1(scr_prog,'uCurveRight',   float(edge.get('right',0)))
    s1(scr_prog,'uContentScale', cs)

    bg_prog['u_color'].value=hex_rgb('#070b12')

    # ── FFmpeg decode ─────────────────────────────────────────
    print('  Starting video decode...')
    dec=subprocess.Popen(
        ['ffmpeg','-i',str(vid),'-vf',f'fps={FPS},scale={VW}:{VH}',
         '-f','rawvideo','-pix_fmt','rgb24','-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fbytes=VW*VH*3

    def next_vid_frame():
        buf=b''
        while len(buf)<fbytes:
            chunk=dec.stdout.read(fbytes-len(buf))
            if not chunk: return False
            buf+=chunk
        vid_tex.write(buf)
        return True

    # ── FFmpeg encode ─────────────────────────────────────────
    print('  Starting NVENC encode...')
    if aud != vid:
        # Mix separate audio file (aud) + video's own audio track (vid)
        enc_cmd=['ffmpeg','-y',
            '-f','rawvideo','-pix_fmt','rgba','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0',
            '-i',str(aud),
            '-i',str(vid),
            '-filter_complex','[1:a][2:a]amix=inputs=2:duration=longest:dropout_transition=2:normalize=0[aout]',
            '-map','0:v','-map','[aout]',
            '-c:v','h264_nvenc','-preset',args.preset,
            '-rc','vbr','-cq','20','-b:v',args.bitrate,
            '-pix_fmt','yuv420p',
            '-c:a','aac','-b:a','320k',
            str(OUT)]
    else:
        enc_cmd=['ffmpeg','-y',
            '-f','rawvideo','-pix_fmt','rgba','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0',
            '-i',str(vid),
            '-map','0:v','-map','1:a',
            '-c:v','h264_nvenc','-preset',args.preset,
            '-rc','vbr','-cq','20','-b:v',args.bitrate,
            '-pix_fmt','yuv420p',
            '-c:a','aac','-b:a','320k',
            str(OUT)]
    enc=subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # ── GL state ──────────────────────────────────────────────
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func=moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    ctx.depth_func='<'

    orb_st=OrbState()
    dt=1.0/FPS

    Mscr_b=mat_bytes(Mscr)
    Mgnd_b=mat_bytes(np.eye(4,dtype='f4'))
    Mbg_b =mat_bytes(Mbg)

    print('  Rendering...\n')
    t0=time.time(); last_log=0.0

    for fi in range(NF):
        t  =fi*dt
        nrg=energies[fi] if fi<len(energies) else 0.0

        orb=orb_st.update(t,nrg,dt)
        ox,oy,oz=orb['lp']
        Morb=(Mg @ translate(ox,oy,oz)
               @ euler_xyz(orb['rx'],orb['ry'],orb['rz'])
               @ scale(orb['scale']))
        orb_wp=tuple((Mg@np.array([ox,oy,oz,1.0],'f4'))[:3])
        Morb_b=mat_bytes(Morb)

        next_vid_frame()

        # Clear
        fbo.clear(*hex_rgb('#070a11'),1.0, depth=1.0)

        # Background (no depth test)
        ctx.disable(moderngl.DEPTH_TEST)
        bg_prog['u_proj'].write(P); bg_prog['u_view'].write(V)
        bg_prog['u_model'].write(Mbg_b)
        bg_vao.render(moderngl.TRIANGLES)
        ctx.enable(moderngl.DEPTH_TEST)

        # Ground
        gnd_prog['u_proj'].write(P); gnd_prog['u_view'].write(V)
        gnd_prog['u_model'].write(Mgnd_b)
        s3(gnd_prog,'u_orb_pos',    orb_wp)
        s1(gnd_prog,'u_orb_intensity', orb['li'])
        t_alb.use(0); t_nrm.use(1); t_rgh.use(2)
        gnd_vao.render(moderngl.TRIANGLES)

        # Screen
        scr_prog['u_proj'].write(P); scr_prog['u_view'].write(V)
        scr_prog['u_model'].write(Mscr_b)
        vid_tex.use(3)
        scr_vao.render(moderngl.TRIANGLES)

        # Orb (transparent)
        orb_prog['u_proj'].write(P); orb_prog['u_view'].write(V)
        orb_prog['u_model'].write(Morb_b)
        orb_prog['u_camera_pos'].value=cam_pos
        orb_prog['uEnergy'].value     =orb['energy']
        orb_prog['uEmitStr'].value    =orb['emit']
        orb_vao.render(moderngl.TRIANGLES)

        # Read pixels (OpenGL = bottom-up → flip for video)
        raw=fbo.read(components=4)
        frame=np.ascontiguousarray(
            np.frombuffer(raw,'u1').reshape(H,W,4)[::-1])
        try:
            enc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print('\n  FFmpeg pipe closed unexpectedly, stopping.')
            break

        now=time.time()
        if now-last_log>2.0 or fi==NF-1:
            elapsed=now-t0
            rfps=(fi+1)/max(elapsed,0.001)
            eta=max(0,(NF-fi-1)/max(rfps,0.001))
            print(f'\r  [{(fi+1)/NF*100:5.1f}%] {fi+1}/{NF} | '
                  f'{rfps:.1f} fps | {elapsed:.0f}s | ETA {eta:.0f}s  ',
                  end='',flush=True)
            last_log=now

    print('\n\n  Finalizing...')
    try: enc.stdin.close()
    except: pass
    enc.wait()
    dec.stdout.close(); dec.wait()
    ctx.release()
    print(f'  Done → {OUT}\n')

if __name__=='__main__':
    main()
