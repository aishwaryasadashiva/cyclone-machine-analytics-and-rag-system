"""Generates synthetic sample PDFs (stand-ins for the Drive corpus, which was not reachable offline)."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
DOCS = {
"Cyclone_Operating_Manual": [
 ("1. Purpose and normal operating window", "The preheater cyclone separates hot kiln feed material from process gas. Normal operation holds the cyclone inlet gas temperature between 850 and 920 C and the outlet gas temperature within 30 C of the inlet temperature. Inlet draft is normally between -150 and -250 mbar; the cone draft is normally 20 to 40 mbar more negative than the outlet draft. Material temperature at the cyclone outlet is typically 900 to 950 C."),
 ("2. Reading the draft signals", "A sudden drop in draft magnitude, meaning the draft moving toward zero, indicates loss of induced-draft fan capacity, a closing damper, or a trip of the fan. If all three drafts move toward zero together, treat it as a fan or damper event and prepare for shutdown. A draft that becomes more negative than normal with falling gas temperature indicates false air ingress, for example an open inspection door or a leaking expansion joint."),
 ("3. Cone draft behaviour", "When the cone draft becomes less negative than the outlet draft by more than 25 mbar, material build-up or a blockage in the cone or dip-leg is likely. Confirm by checking the cone for pressure fluctuation and trigger the air cannons. Persistent divergence for more than 30 minutes requires a poke-hole inspection."),
 ("4. Temperature relationships", "Inlet and outlet gas temperatures are strongly correlated. An outlet temperature that falls far below the inlet temperature while drafts stay normal indicates heat loss, usually from false air or a failed outlet thermocouple. A material temperature reading that is flat at about 40 C while gas temperatures are normal indicates an instrument fault, not a process upset."),
],
"Startup_and_Shutdown_SOP": [
 ("1. Planned shutdown", "Reduce feed and fuel gradually, then stop the induced-draft fan only after the kiln feed has been cleared. During cooling the inlet gas temperature decays toward ambient over several hours while the drafts collapse to near zero. Record the shutdown start time and reason in the log."),
 ("2. Restart after a shutdown", "Start the induced-draft fan first and confirm the drafts have returned to the normal negative range before firing. The first hour after restart is an unstable period: the inlet temperature ramps upward at more than 25 C per hour and temperature variance is high. Do not raise feed until the inlet temperature exceeds 800 C and the 1-hour variability falls."),
 ("3. Unplanned trips", "A trip is recognised by all drafts rising toward zero within minutes. Short stops of under two hours may be restarted directly if no blockage is suspected. Longer outages require inspection of the cyclone cone and the ID fan before restart."),
 ("4. Idle periods", "During idle periods the temperature sensors may report constant low values and communication errors such as I/O Timeout or Not Connect. These readings must not be used for process control or reporting."),
],
"Troubleshooting_Guide": [
 ("Symptom: sudden draft drop", "A sudden draft drop indicates loss of suction. Check in order: ID fan speed and trip status, damper position, blockage in the downstream duct, and upstream surge. If inlet temperature also falls, a shutdown is likely within the hour; alert the shift supervisor."),
 ("Symptom: rapid fall of inlet gas temperature", "A rapid fall of the inlet gas temperature without a draft change points to a drop in upstream heat input such as reduced firing. With a draft that becomes more negative the likely cause is cold false air entering the system."),
 ("Symptom: erratic or high-variance readings", "High rolling variance of inlet temperature or draft indicates an unstable operating regime such as startup, kiln upset, or an unstable feed. Persistent variance outside startup should be investigated for sticking dampers or a coating fall."),
 ("Symptom: frozen sensor value", "A sensor that stays at exactly the same value while others move is stuck or stale. Check the transmitter, the wiring, and the DCS signal quality flag. Replace or recalibrate if the fault persists."),
],
"Alarm_and_Monitoring_Guide": [
 ("1. Recommended alarm rules", "Raise a warning when the 15 minute change in inlet draft exceeds 40 mbar toward zero. Raise a high alarm when the inlet temperature falls more than 30 C in 15 minutes together with a draft rise. Raise an information alert when the one hour standard deviation of inlet temperature exceeds 45 C outside the first hour after restart."),
 ("2. Data quality monitoring", "Monitor communication error codes. More than three consecutive Not Connect or I/O Timeout samples must raise a data quality alert. Values pinned at instrument limits such as 1375 C indicate sensor saturation."),
],
"Safety_Procedures": [
 ("1. Entry into the cyclone", "Cyclone and preheater tower entry is a confined space activity requiring a permit, lock-out tag-out of the fan and feeders, gas testing, and a standby person. Never enter while the unit is hot."),
 ("2. Hot material hazards", "Blockage clearing with air cannons or poking can release hot material. Operators must wear heat-resistant clothing, face shield and stay clear of the discharge path. Personnel must be evacuated from the area below the cyclone before any rapid clearing."),
 ("3. Emergency response", "In case of burn injury, fire or gas release, press the emergency stop, evacuate, and call the site emergency number. Do not attempt to restart the unit until the safety officer clears the area."),
],
}
if __name__ == "__main__":
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs"); os.makedirs(here, exist_ok=True)
    st = getSampleStyleSheet()
    for name, secs in DOCS.items():
        doc = SimpleDocTemplate(os.path.join(here, name + ".pdf"), pagesize=A4)
        flow = [Paragraph(name.replace("_", " "), st["Title"]), Spacer(1, 12)]
        for i, (h, t) in enumerate(secs):
            flow += [Paragraph(h, st["Heading2"]), Paragraph(t, st["BodyText"]), Spacer(1, 10)]
            if i == 1: flow.append(PageBreak())
        doc.build(flow)
    print("wrote", len(DOCS), "PDFs to", here)
