#!/bin/bash
# Wrapper script to launch Robocode GUI with forced software rendering
# This ensures the Java 2D options are definitely applied at launch time

# Kill any _JAVA_OPTIONS that might conflict
unset _JAVA_OPTIONS

# Force pure software rendering pipeline - all GPU acceleration disabled
# Added aggressive flags for canvas/game rendering
export JAVA_TOOL_OPTIONS="-Djava.awt.headless=false \
    -Dsun.java2d.opengl=false \
    -Dsun.java2d.xrender=false \
    -Dsun.java2d.d3d=false \
    -Dsun.java2d.noddraw=true \
    -Dsun.java2d.pmoffscreen=false \
    -Dsun.awt.disableMixing=true \
    -Dsun.java2d.renderer=sun.java2d.marlin.MarlinRenderingEngine"

# Force X11 toolkit
export AWT_TOOLKIT=XToolkit

# Ensure DISPLAY is set
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:99
fi

echo "[gui_wrapper] Launching Robocode GUI on DISPLAY=$DISPLAY"
echo "[gui_wrapper] JAVA_TOOL_OPTIONS=$JAVA_TOOL_OPTIONS"

# Launch the GUI jar with explicit options for BOTH swing and canvas rendering
exec java -Xmx1024m \
    -Djava.net.preferIPv4Stack=true \
    -Djava.security.egd=file:/dev/./urandom \
    -Djava.awt.headless=false \
    -Dsun.java2d.opengl=false \
    -Dsun.java2d.xrender=false \
    -Dsun.java2d.d3d=false \
    -Dsun.java2d.noddraw=true \
    -Dsun.java2d.pmoffscreen=false \
    -Dsun.java2d.renderer=sun.java2d.marlin.MarlinRenderingEngine \
    -Dawt.useSystemAAFontSettings=off \
    -Dswing.defaultlaf=javax.swing.plaf.metal.MetalLookAndFeel \
    -Dsun.awt.noerasebackground=false \
    -Dsun.awt.erasebackgroundonresize=true \
    -jar /robocode/gui.jar "$@"
