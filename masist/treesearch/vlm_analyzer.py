"""
VLM Analyzer for Phase 6
Analyzes experiment plots using Vision-Language Model (GPT-4o)
"""
import base64
import logging
import os
from typing import cast, Any

from .journal import Node
from .backend import query
from .utils.function_specs import vlm_feedback_spec, plot_selection_spec

logger = logging.getLogger(__name__)


class VLMAnalyzer:
    """Analyze experimental plots using Vision-Language Model"""

    def __init__(self, task_desc: str, cfg: Any):
        self.task_desc = task_desc
        self.cfg = cfg

    def analyze(self, node: Node) -> None:
        """
        Analyze experimental plots using VLM
        Equivalent to _analyze_plots_with_vlm() from parallel_agent.py
        """
        if not node.plot_paths:
            logger.warning(f"No plot paths found for node {node.id}")
            return

        logger.info(f"Analyzing {len(node.plot_paths)} plots for node {node.id}")
        print(f"[cyan]Plot paths:[/cyan] {node.plot_paths}")

        # Step 1: Select plots if there are too many (>10)
        selected_plots = self._select_plots(node)

        # Step 2: Encode images to base64
        user_message = self._build_vlm_message(selected_plots)

        # Step 3: Query VLM for analysis
        response = cast(
            dict,
            query(
                system_message=None,
                user_message=user_message,
                func_spec=vlm_feedback_spec,
                model=self.cfg.agent.vlm_feedback.model,
                temperature=self.cfg.agent.vlm_feedback.temp,
            ),
        )

        print(
            f"[cyan]VLM response from {self.cfg.agent.vlm_feedback.model}:[/cyan] {response}"
        )

        # Step 4: Store analysis results in node
        if response["valid_plots_received"]:
            node.is_buggy_plots = False
        else:
            node.is_buggy_plots = True

        # Add plot paths to each analysis entry
        for index, analysis in enumerate(response["plot_analyses"]):
            if index < len(node.plot_paths):
                analysis["plot_path"] = node.plot_paths[index]

        node.plot_analyses = response["plot_analyses"]
        node.vlm_feedback_summary = response["vlm_feedback_summary"]

        logger.info(f"VLM analysis completed for node {node.id}")

    def _select_plots(self, node: Node) -> list[str]:
        """
        Select plots to analyze. If >10 plots, use LLM to select most relevant ones.
        """
        if len(node.plot_paths) <= 10:
            return node.plot_paths

        logger.warning(
            f"{len(node.plot_paths)} plots received, calling LLM to select 10 most relevant"
        )
        print(
            f"[red]Warning: {len(node.plot_paths)} plots received, this may be too many to analyze effectively. "
            f"Calling LLM to select the most relevant plots to analyze.[/red]"
        )

        # Prompt LLM to select 10 plots
        prompt_select_plots = {
            "Introduction": (
                "You are an experienced AI researcher analyzing experimental results. "
                "You have been provided with plots from a machine learning experiment. "
                "Please select 10 most relevant plots to analyze. "
                "For similar plots (e.g. generated samples at each epoch), select only at most 5 plots at a suitable interval of epochs."
                "Format your response as a list of plot paths, where each plot path includes the full path to the plot file."
            ),
            "Plot paths": node.plot_paths,
        }

        try:
            response_select_plots = cast(
                dict,
                query(
                    system_message=prompt_select_plots,
                    user_message=None,
                    func_spec=plot_selection_spec,
                    model=self.cfg.agent.feedback.model,
                    temperature=self.cfg.agent.feedback.temp,
                ),
            )

            print(f"[cyan]Plot selection response:[/cyan] {response_select_plots}")
            selected_plots = response_select_plots.get("selected_plots", [])

            # Validate that all paths exist and are image files
            valid_plots = []
            for plot_path in selected_plots:
                if (
                    isinstance(plot_path, str)
                    and os.path.exists(plot_path)
                    and plot_path.lower().endswith((".png", ".jpg", ".jpeg"))
                ):
                    valid_plots.append(plot_path)
                else:
                    logger.warning(f"Invalid plot path received: {plot_path}")

            if valid_plots:
                print(f"[cyan]Selected valid plots:[/cyan] {valid_plots}")
                return valid_plots
            else:
                logger.warning(
                    "No valid plot paths found in response, falling back to first 10 plots"
                )
                return self._fallback_plot_selection(node)

        except Exception as e:
            logger.error(
                f"Error in plot selection: {str(e)}; falling back to first 10 plots"
            )
            return self._fallback_plot_selection(node)

    def _fallback_plot_selection(self, node: Node) -> list[str]:
        """Fallback: validate and return first 10 plots"""
        selected_plots = []
        for plot_path in node.plot_paths[:10]:
            if os.path.exists(plot_path) and plot_path.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                selected_plots.append(plot_path)
            else:
                logger.warning(f"Invalid plot path in fallback: {plot_path}")
        return selected_plots

    def _build_vlm_message(self, selected_plots: list[str]) -> list[dict]:
        """Build VLM message with text prompt and base64-encoded images"""
        print("[cyan]Encoding images to base64[/cyan]")

        user_message = [
            {
                "type": "text",
                "text": (
                    "You are an experienced AI researcher analyzing experimental results. "
                    "You have been provided with plots from a machine learning experiment. "
                    f"This experiment is based on the following research idea: {self.task_desc}"
                    "Please analyze these plots and provide detailed insights about the results. "
                    "If you don't receive any plots, say 'No plots received'. "
                    "Never make up plot analysis. "
                    "Please return the analyzes with strict order of uploaded images, but DO NOT include any word "
                    "like 'the first plot'."
                ),
            }
        ]

        # Add images
        for plot_path in selected_plots:
            encoded_image = self._encode_image_to_base64(plot_path)
            if encoded_image:
                user_message.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    }
                )

        return user_message

    def _encode_image_to_base64(self, image_path: str) -> str | None:
        """Encode image file to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
