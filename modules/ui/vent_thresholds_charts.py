"""
Vent Thresholds — CPET chart panels (VE/VO2, VE/VCO2, RER, VE-only) using Matplotlib.
"""
import streamlit as st
import pandas as pd


def render_cpet_charts(cpet_result: dict) -> None:
    """
    Render the "📊 Wykresy CPET" expander with all chart panels.

    Draws gas-exchange panels (VE/VO2, VE/VCO2, RER) when available,
    falls back to a simple VE vs Power chart in VE-only mode.
    Also renders the secondary metabolic zones table and raw step data expander.
    """
    st.markdown("---")
    with st.expander("📊 Wykresy CPET", expanded=True):
        import matplotlib.pyplot as plt

        df_s = cpet_result.get("df_steps")
        v1_w = cpet_result.get("vt1_watts")
        v2_w = cpet_result.get("vt2_watts")

        if df_s is None or len(df_s) == 0:
            st.warning("Brak danych schodków do wyświetlenia wykresów")
            return

        has_ve_vo2 = "ve_vo2" in df_s.columns and df_s["ve_vo2"].notna().any()
        has_ve_vco2 = "ve_vco2" in df_s.columns and df_s["ve_vco2"].notna().any()

        if has_ve_vo2 or has_ve_vco2:
            st.markdown("### Wykresy Ekwiwalentów Wentylacyjnych")

            n_panels = 2 if (has_ve_vo2 and has_ve_vco2) else 1
            fig, axes = plt.subplots(1, n_panels, figsize=(14, 5))
            plt.style.use("dark_background")
            fig.patch.set_facecolor("#0E1117")

            if not isinstance(axes, (list, tuple)):
                axes = [axes]
            ax_idx = 0

            if has_ve_vo2:
                ax1 = axes[ax_idx]
                ax1.set_facecolor("#0E1117")
                ax1.plot(
                    df_s["power"],
                    df_s["ve_vo2"],
                    "b-o",
                    linewidth=2,
                    markersize=6,
                    label="VE/VO2",
                )
                ax1.set_xlabel("Moc [W]", color="white")
                ax1.set_ylabel("VE/VO2", color="#5da5da")
                ax1.set_title("VE/VO2 vs Power (VT1 Detection)", color="white", pad=10)
                ax1.grid(True, alpha=0.2)

                if v1_w:
                    ax1.axvline(
                        v1_w,
                        color="#ffa15a",
                        linestyle="--",
                        linewidth=2,
                        label=f"VT1: {v1_w}W",
                    )
                    vt1_row = df_s[df_s["power"] == v1_w]
                    if len(vt1_row) > 0:
                        y_vt1 = vt1_row["ve_vo2"].iloc[0]
                        ax1.scatter([v1_w], [y_vt1], color="#ffa15a", s=150, zorder=5, marker="*")

                ax1.legend(loc="upper left")
                ax_idx += 1

            if has_ve_vco2:
                ax2 = axes[ax_idx] if ax_idx < len(axes) else axes[0]
                ax2.set_facecolor("#0E1117")
                ax2.plot(
                    df_s["power"],
                    df_s["ve_vco2"],
                    "r-o",
                    linewidth=2,
                    markersize=6,
                    label="VE/VCO2",
                )
                ax2.set_xlabel("Moc [W]", color="white")
                ax2.set_ylabel("VE/VCO2", color="#ef553b")
                ax2.set_title("VE/VCO2 vs Power (VT2 Detection)", color="white", pad=10)
                ax2.grid(True, alpha=0.2)

                if v1_w:
                    ax2.axvline(
                        v1_w,
                        color="#ffa15a",
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.7,
                        label=f"VT1: {v1_w}W",
                    )
                if v2_w:
                    ax2.axvline(
                        v2_w,
                        color="#ef553b",
                        linestyle="--",
                        linewidth=2,
                        label=f"VT2: {v2_w}W",
                    )
                    vt2_row = df_s[df_s["power"] == v2_w]
                    if len(vt2_row) > 0:
                        y_vt2 = vt2_row["ve_vco2"].iloc[0]
                        ax2.scatter([v2_w], [y_vt2], color="#ef553b", s=150, zorder=5, marker="*")

                ax2.legend(loc="upper left")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # RER chart
            if "rer" in df_s.columns and df_s["rer"].notna().any():
                st.markdown("### RER (Respiratory Exchange Ratio)")
                fig_rer, ax_rer = plt.subplots(figsize=(10, 4))
                ax_rer.set_facecolor("#0E1117")
                fig_rer.patch.set_facecolor("#0E1117")

                ax_rer.plot(
                    df_s["power"], df_s["rer"], "g-o", linewidth=2, markersize=6, label="RER"
                )
                ax_rer.axhline(
                    1.0, color="yellow", linestyle="--", linewidth=1.5, alpha=0.7, label="RER = 1.0"
                )
                ax_rer.set_xlabel("Moc [W]", color="white")
                ax_rer.set_ylabel("RER (VCO2/VO2)", color="#60bd68")
                ax_rer.set_title("Respiratory Exchange Ratio vs Power", color="white", pad=10)
                ax_rer.grid(True, alpha=0.2)

                if v2_w:
                    ax_rer.axvline(
                        v2_w,
                        color="#ef553b",
                        linestyle="--",
                        linewidth=2,
                        label=f"VT2: {v2_w}W",
                    )

                ax_rer.legend(loc="upper left")
                plt.tight_layout()
                st.pyplot(fig_rer)
                plt.close(fig_rer)

        else:
            # VE-only mode chart
            st.markdown("### Wykres VE vs Power")

            fig, ax1 = plt.subplots(figsize=(10, 5))
            plt.style.use("dark_background")
            fig.patch.set_facecolor("#0E1117")
            ax1.set_facecolor("#0E1117")

            if "ve" in df_s.columns:
                ax1.plot(df_s["power"], df_s["ve"], "b-o", linewidth=2, label="VE (L/min)")
            elif "ve_smooth" in df_s.columns:
                ax1.plot(
                    df_s["power"], df_s["ve_smooth"], "b-o", linewidth=2, label="VE (L/min)"
                )

            ax1.set_xlabel("Moc [W]", color="white")
            ax1.set_ylabel("Wentylacja [L/min]", color="#5da5da")

            if v1_w:
                ax1.axvline(
                    v1_w, color="#ffa15a", linestyle="--", linewidth=2, label=f"VT1: {v1_w}W"
                )
            if v2_w:
                ax1.axvline(
                    v2_w, color="#ef553b", linestyle="--", linewidth=2, label=f"VT2: {v2_w}W"
                )

            ax1.set_title("VE vs Power z Progami VT1/VT2", color="white", pad=10)
            ax1.grid(True, alpha=0.2)
            ax1.legend(loc="upper left")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Secondary zones table
        st.markdown("### 🎯 Strefy Metaboliczne")
        if v1_w and v2_w:
            zones_data = [
                {
                    "Strefa": "Z1 (Recovery)",
                    "Zakres": f"< {v1_w} W",
                    "Opis": "Regeneracja, rozgrzewka",
                    "Metabolizm": "100% Tlenowy",
                },
                {
                    "Strefa": "Z2 (Endurance)",
                    "Zakres": f"{v1_w} - {int((v1_w + v2_w) / 2)} W",
                    "Opis": "Baza tlenowa",
                    "Metabolizm": "Dominująco tlenowy",
                },
                {
                    "Strefa": "Z3 (Tempo)",
                    "Zakres": f"{int((v1_w + v2_w) / 2)} - {v2_w} W",
                    "Opis": "Sweet Spot",
                    "Metabolizm": "Mieszany",
                },
                {
                    "Strefa": "Z4 (Threshold)",
                    "Zakres": f"{v2_w} - {int(v2_w * 1.05)} W",
                    "Opis": "FTP, MLSS",
                    "Metabolizm": "Glikolityczny",
                },
                {
                    "Strefa": "Z5 (VO2max)",
                    "Zakres": f"> {int(v2_w * 1.05)} W",
                    "Opis": "Interwały",
                    "Metabolizm": "Anaerobowy",
                },
            ]
            st.dataframe(pd.DataFrame(zones_data), hide_index=True, use_container_width=True)
        else:
            st.warning("Brak danych do wygenerowania stref metabolicznych")

        # Raw step data
        with st.expander("📝 Dane schodków (raw)", expanded=False):
            display_cols = ["step", "power", "ve"]
            for col in ["vo2", "vco2", "ve_vo2", "ve_vco2", "rer", "hr"]:
                if col in df_s.columns:
                    display_cols.append(col)

            available_cols = [c for c in display_cols if c in df_s.columns]
            st.dataframe(df_s[available_cols].round(2), use_container_width=True)
