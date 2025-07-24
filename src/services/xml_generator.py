"""
Adobe Premiere XML Generation Service.

This service generates XMEML format files for Adobe Premiere Pro,
creating timeline markers and clip sequences based on editing decisions.
"""

import logging
from typing import Dict, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom

from ..models.video import VideoInfo
from ..models.editing import EditingResult
from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class XMLGeneratorError(Exception):
    """Custom exception for XML generation errors."""
    pass


class XMLGenerator:
    """
    Adobe Premiere XML generator service.
    
    Creates XMEML format files compatible with Adobe Premiere Pro,
    including timeline markers, clip sequences, and media references.
    """
    
    def __init__(self):
        """Initialize the XML generator."""
        self.settings = get_settings()
    
    def generate_premiere_xml(self, video_info: VideoInfo, 
                             editing_result: EditingResult,
                             video_path: str) -> str:
        """
        Generate Adobe Premiere XML from editing results.
        
        Args:
            video_info: Information about the source video
            editing_result: AI editing results with selected segments
            video_path: Path to the source video file
            
        Returns:
            XML content as string
            
        Raises:
            XMLGeneratorError: If XML generation fails
        """
        try:
            # Create root XMEML element
            # Não adicionar DOCTYPE aqui, será adicionado pelo _format_xml
            root = ET.Element("xmeml")
            root.set("version", "4")
            
            # Create project element
            project = ET.SubElement(root, "project")
            
            # Add project name
            name = ET.SubElement(project, "name")
            name.text = f"AI_Edit_{video_info.filename}"
            
            # Create children element
            children = ET.SubElement(project, "children")
            
            # Add bins (folders) com master clip
            self._add_bins(children, video_info, video_path)
            
            # Add sequence
            self._add_sequence(children, video_info, editing_result, video_path)
            
            # Format and return XML
            return self._format_xml(root)
            
        except Exception as e:
            logger.error(f"Error generating Premiere XML: {str(e)}")
            raise XMLGeneratorError(f"Failed to generate XML: {str(e)}")
    
    def _add_bins(self, parent: ET.Element, video_info: VideoInfo, video_path: str) -> None:
        """
        Add project bins (folders) to the XML.
        
        Args:
            parent: Parent XML element
            video_info: Video information
            video_path: Path to video file
        """
        # Create media bin
        media_bin = ET.SubElement(parent, "bin")
        
        # Add bin properties
        name = ET.SubElement(media_bin, "name")
        name.text = "Media"
        
        # Add children element for bin
        children = ET.SubElement(media_bin, "children")
        
        # Add clip element (estrutura correta baseada no XML que funcionou)
        clip_elem = ET.SubElement(children, "clip")
        clip_elem.set("id", "master_clip")
        
        # Add clip name
        clip_name = ET.SubElement(clip_elem, "name")
        clip_name.text = video_info.filename
        
        # Add clip duration
        clip_duration = ET.SubElement(clip_elem, "duration")
        clip_duration.text = str(int(video_info.duration * video_info.fps))
        
        # Add clip rate
        clip_rate = ET.SubElement(clip_elem, "rate")
        clip_timebase = ET.SubElement(clip_rate, "timebase")
        clip_timebase.text = str(int(video_info.fps))
        clip_ntsc = ET.SubElement(clip_rate, "ntsc")
        clip_ntsc.text = "FALSE"
        
        # Add file element inside clip
        file_elem = ET.SubElement(clip_elem, "file")
        file_elem.set("id", "master_media_file")
        
        # Add file name
        file_name = ET.SubElement(file_elem, "name")
        file_name.text = video_info.filename
        
        # Add path URL - formato correto para o Premiere
        pathurl = ET.SubElement(file_elem, "pathurl")
        formatted_path = video_path.replace('\\', '/')
        pathurl.text = f"file:///{formatted_path}"
        
        # Add file rate
        file_rate = ET.SubElement(file_elem, "rate")
        file_timebase = ET.SubElement(file_rate, "timebase")
        file_timebase.text = str(int(video_info.fps))
        file_ntsc = ET.SubElement(file_rate, "ntsc")
        file_ntsc.text = "FALSE"
        
        # Add file duration
        file_duration = ET.SubElement(file_elem, "duration")
        file_duration.text = str(int(video_info.duration * video_info.fps))
        
        # Add media element (estrutura simplificada como no XML que funcionou)
        media = ET.SubElement(file_elem, "media")
        
        # Add video track (estrutura simples)
        video = ET.SubElement(media, "video")
        ET.SubElement(video, "track")
        
        # Add audio track if present (estrutura simples)
        if video_info.has_audio:
            audio = ET.SubElement(media, "audio")
            ET.SubElement(audio, "track")
        
        # Add clips bin
        clips_bin = ET.SubElement(parent, "bin")
        clips_name = ET.SubElement(clips_bin, "name")
        clips_name.text = "AI Edited Clips"
        
        ET.SubElement(clips_bin, "children")
    
    # Método _add_media_files foi removido pois sua funcionalidade foi incorporada ao _add_bins
    # que agora cria o arquivo master_media_file com todas as propriedades necessárias
    
    def _add_video_track_properties(self, video_track: ET.Element, 
                                   video_info: VideoInfo) -> None:
        """
        Add video track properties to the XML.
        
        Args:
            video_track: Video track XML element
            video_info: Video information
        """
        # Add sample characteristics
        samplecharacteristics = ET.SubElement(video_track, "samplecharacteristics")
        
        # Add rate (frame rate)
        rate = ET.SubElement(samplecharacteristics, "rate")
        timebase = ET.SubElement(rate, "timebase")
        timebase.text = str(int(video_info.fps))
        ntsc = ET.SubElement(rate, "ntsc")
        ntsc.text = "TRUE" if video_info.fps == 29.97 else "FALSE"
        
        # Add resolution
        width_elem = ET.SubElement(samplecharacteristics, "width")
        height_elem = ET.SubElement(samplecharacteristics, "height")
        
        if 'x' in video_info.resolution:
            width, height = video_info.resolution.split('x')
            width_elem.text = width.strip()
            height_elem.text = height.strip()
        else:
            width_elem.text = "1920"
            height_elem.text = "1080"
        
        # Add other properties
        anamorphic = ET.SubElement(samplecharacteristics, "anamorphic")
        anamorphic.text = "FALSE"
        
        pixelaspectratio = ET.SubElement(samplecharacteristics, "pixelaspectratio")
        pixelaspectratio.text = "square"
        
        fielddominance = ET.SubElement(samplecharacteristics, "fielddominance")
        fielddominance.text = "none"
    
    def _add_audio_track_properties(self, audio_track: ET.Element, 
                                   video_info: VideoInfo) -> None:
        """
        Add audio track properties to the XML.
        
        Args:
            audio_track: Audio track XML element
            video_info: Video information
        """
        # Add sample characteristics
        samplecharacteristics = ET.SubElement(audio_track, "samplecharacteristics")
        
        # Add sample rate
        samplerate = ET.SubElement(samplecharacteristics, "samplerate")
        samplerate.text = str(self.settings.audio_sample_rate)
        
        # Add channel count
        channelcount = ET.SubElement(samplecharacteristics, "channelcount")
        channelcount.text = "2"  # Stereo
        
        # Add bit depth
        depth = ET.SubElement(samplecharacteristics, "depth")
        depth.text = "16"
    
    def _add_sequence(self, parent: ET.Element, video_info: VideoInfo,
                     editing_result: EditingResult, video_path: str) -> None:
        """
        Add sequence with edited clips to the XML.
        
        Args:
            parent: Parent XML element
            video_info: Video information
            editing_result: Editing results
            video_path: Path to video file
        """
        # Create sequence element
        sequence = ET.SubElement(parent, "sequence")
        sequence.set("id", "ai_sequence")
        
        # Add sequence properties
        name = ET.SubElement(sequence, "name")
        name.text = f"{video_info.filename}_AI_Cuts"
        
        # Add duration
        duration = ET.SubElement(sequence, "duration")
        duration.text = str(int(editing_result.final_duration * video_info.fps))
        
        # Add rate
        rate = ET.SubElement(sequence, "rate")
        timebase = ET.SubElement(rate, "timebase")
        timebase.text = str(int(video_info.fps))
        ntsc = ET.SubElement(rate, "ntsc")
        ntsc.text = "TRUE" if video_info.fps == 29.97 else "FALSE"
        
        # Add timecode
        timecode = ET.SubElement(sequence, "timecode")
        tc_rate = ET.SubElement(timecode, "rate")
        tc_timebase = ET.SubElement(tc_rate, "timebase")
        tc_timebase.text = str(int(video_info.fps))
        tc_ntsc = ET.SubElement(tc_rate, "ntsc")
        tc_ntsc.text = "TRUE" if video_info.fps == 29.97 else "FALSE"
        
        # Add string timecode
        tc_string = ET.SubElement(timecode, "string")
        tc_string.text = "00:00:00:00"
        
        # Add frame
        tc_frame = ET.SubElement(timecode, "frame")
        tc_frame.text = "0"
        
        # Add display format
        tc_displayformat = ET.SubElement(timecode, "displayformat")
        tc_displayformat.text = "NDF"
        
        # Add media
        media = ET.SubElement(sequence, "media")
        
        # Add video track
        video_track = ET.SubElement(media, "video")
        
        # Add video format
        format_elem = ET.SubElement(video_track, "format")
        sample_chars = ET.SubElement(format_elem, "samplecharacteristics")
        
        # Add rate
        rate = ET.SubElement(sample_chars, "rate")
        timebase = ET.SubElement(rate, "timebase")
        timebase.text = str(int(video_info.fps))
        ntsc = ET.SubElement(rate, "ntsc")
        ntsc.text = "TRUE" if video_info.fps == 29.97 else "FALSE"
        
        # Add resolution
        if hasattr(video_info, "resolution") and video_info.resolution:
            width_height = video_info.resolution.split("x")
            if len(width_height) == 2:
                width = ET.SubElement(sample_chars, "width")
                width.text = width_height[0]
                height = ET.SubElement(sample_chars, "height")
                height.text = width_height[1]
            else:
                # Default HD resolution
                width = ET.SubElement(sample_chars, "width")
                width.text = "1920"
                height = ET.SubElement(sample_chars, "height")
                height.text = "1080"
        else:
            # Default HD resolution
            width = ET.SubElement(sample_chars, "width")
            width.text = "1920"
            height = ET.SubElement(sample_chars, "height")
            height.text = "1080"
        
        # Add pixel aspect ratio
        pixel_aspect = ET.SubElement(sample_chars, "pixelaspectratio")
        pixel_aspect.text = "square"
        
        # Add track for clips
        track = ET.SubElement(video_track, "track")
        
        # Add video clips
        self._add_video_clips(track, video_info, editing_result, video_path)
        
        # Add audio track if present
        if video_info.has_audio:
            audio_track = ET.SubElement(media, "audio")
            # Add track for clips
            track = ET.SubElement(audio_track, "track")
            # Add audio clips
            self._add_audio_clips(track, video_info, editing_result, video_path)
    
    def _add_video_clips(self, video_track: ET.Element, video_info: VideoInfo,
                        editing_result: EditingResult, video_path: str) -> None:
        """
        Add video clip items to the track.
        
        Args:
            video_track: Video track XML element
            video_info: Video information
            editing_result: Editing results
            video_path: Path to video file
        """
        # O track já é passado como parâmetro, não precisamos criar um novo
        
        # Get selected segments from editing context
        selected_segments = []
        
        # Usar timestamps reais dos segmentos selecionados
        if not editing_result.decisions:
            # Fallback: usar todo o vídeo
            selected_segments.append({
                'id': 'full_video',
                'start': 0.0,
                'end': video_info.duration,
                'duration': video_info.duration
            })
        else:
            # Usar timestamps reais das decisões
            for decision in editing_result.decisions:
                if decision.decision_type == "keep" and decision.start_time is not None and decision.end_time is not None:
                    selected_segments.append({
                        'id': decision.segment_id,
                        'start': decision.start_time,  # Tempo real no vídeo original
                        'end': decision.end_time,      # Tempo real no vídeo original
                        'duration': decision.end_time - decision.start_time
                    })
        
        # Add clip items
        timeline_position = 0
        for i, segment in enumerate(selected_segments):
            clipitem = ET.SubElement(video_track, "clipitem")
            clipitem.set("id", f"clipitem-{i+1}")
            
            # Add clip properties
            name = ET.SubElement(clipitem, "name")
            name.text = f"Clip {i+1}"
            
            # Add enabled property
            enabled = ET.SubElement(clipitem, "enabled")
            enabled.text = "TRUE"
            
            # Add duration
            clip_duration = int((segment['end'] - segment['start']) * video_info.fps)
            duration = ET.SubElement(clipitem, "duration")
            duration.text = str(clip_duration)
            
            # Add start/end points (timeline position)
            start = ET.SubElement(clipitem, "start")
            start.text = str(int(timeline_position * video_info.fps))
            
            end = ET.SubElement(clipitem, "end")
            end.text = str(int((timeline_position + segment['duration']) * video_info.fps))
            
            # Add in/out points (source media)
            in_point = ET.SubElement(clipitem, "in")
            in_point.text = str(int(segment['start'] * video_info.fps))
            
            out_point = ET.SubElement(clipitem, "out")
            out_point.text = str(int(segment['end'] * video_info.fps))
            
            # Add file reference
            file_ref = ET.SubElement(clipitem, "file")
            file_ref.set("id", "master_media_file")
            
            # Add source track (obrigatório para o Premiere)
            sourcetrack = ET.SubElement(clipitem, "sourcetrack")
            mediatype = ET.SubElement(sourcetrack, "mediatype")
            mediatype.text = "video"
            trackindex = ET.SubElement(sourcetrack, "trackindex")
            trackindex.text = "1"
            
            # Add comments with reason and relevance
            comments = ET.SubElement(clipitem, "comments")
            reason = segment.get('reason', 'Este trecho foi selecionado pela IA com base na relevância do conteúdo.')
            relevance = segment.get('relevance', '8/10')
            comments.text = f"Resumo: N/A\nRazão: {reason}\nRelevância: {relevance}"
            
            # Add link to audio clip
            link = ET.SubElement(clipitem, "link")
            linkclipref = ET.SubElement(link, "linkclipref")
            linkclipref.text = f"audioclip-{i+1}"
            mediatype = ET.SubElement(link, "mediatype")
            mediatype.text = "audio"
            trackindex = ET.SubElement(link, "trackindex")
            trackindex.text = "1"
            clipindex = ET.SubElement(link, "clipindex")
            clipindex.text = "1"
            
            # Update timeline position (usando a duração do segmento em segundos, não em frames)
            timeline_position += segment['duration']
    
    def _add_audio_clips(self, audio_track: ET.Element, video_info: VideoInfo,
                        editing_result: EditingResult, video_path: str) -> None:
        """
        Add audio clip items to the track.
        
        Args:
            audio_track: Audio track XML element
            video_info: Video information
            editing_result: Editing results
            video_path: Path to video file
        """
        # O track já é passado como parâmetro, não precisamos criar um novo
        
        # Get selected segments from editing context (mesmos segmentos do vídeo)
        selected_segments = []
        
        # Usar timestamps reais dos segmentos selecionados
        if not editing_result.decisions:
            # Fallback: usar todo o vídeo
            selected_segments.append({
                'id': 'full_video',
                'start': 0.0,
                'end': video_info.duration,
                'duration': video_info.duration
            })
        else:
            # Usar timestamps reais das decisões
            for decision in editing_result.decisions:
                if decision.decision_type == "keep" and decision.start_time is not None and decision.end_time is not None:
                    selected_segments.append({
                        'id': decision.segment_id,
                        'start': decision.start_time,  # Tempo real no vídeo original
                        'end': decision.end_time,      # Tempo real no vídeo original
                        'duration': decision.end_time - decision.start_time
                    })
        
        # Add clip items - sincronizados com os clipes de vídeo
        timeline_position = 0
        for i, segment in enumerate(selected_segments):
            clipitem = ET.SubElement(audio_track, "clipitem")
            clipitem.set("id", f"audioclip-{i+1}")
            
            # Add clip properties
            name = ET.SubElement(clipitem, "name")
            name.text = f"Clip {i+1}"
            
            # Add enabled property
            enabled = ET.SubElement(clipitem, "enabled")
            enabled.text = "TRUE"
            
            # Add duration
            clip_duration = int((segment['end'] - segment['start']) * video_info.fps)
            duration = ET.SubElement(clipitem, "duration")
            duration.text = str(clip_duration)
            
            # Add start/end points (timeline position)
            start = ET.SubElement(clipitem, "start")
            start.text = str(int(timeline_position * video_info.fps))
            
            end = ET.SubElement(clipitem, "end")
            end.text = str(int((timeline_position + segment['duration']) * video_info.fps))
            
            # Add in/out points (source media)
            in_point = ET.SubElement(clipitem, "in")
            in_point.text = str(int(segment['start'] * video_info.fps))
            
            out_point = ET.SubElement(clipitem, "out")
            out_point.text = str(int(segment['end'] * video_info.fps))
            
            # Add file reference
            file_ref = ET.SubElement(clipitem, "file")
            file_ref.set("id", "master_media_file")
            
            # Add source track
            sourcetrack = ET.SubElement(clipitem, "sourcetrack")
            mediatype = ET.SubElement(sourcetrack, "mediatype")
            mediatype.text = "audio"
            trackindex = ET.SubElement(sourcetrack, "trackindex")
            trackindex.text = "1"
            
            # Add link to video clip
            link = ET.SubElement(clipitem, "link")
            linkclipref = ET.SubElement(link, "linkclipref")
            linkclipref.text = f"clipitem-{i+1}"
            mediatype = ET.SubElement(link, "mediatype")
            mediatype.text = "video"
            trackindex = ET.SubElement(link, "trackindex")
            trackindex.text = "1"
            clipindex = ET.SubElement(link, "clipindex")
            clipindex.text = "1"
            
            # Update timeline position
            timeline_position += segment['duration']
    
    def _format_xml(self, root: ET.Element) -> str:
        """
        Format XML with proper indentation and DOCTYPE for Premiere.
        
        Args:
            root: Root XML element
            
        Returns:
            Formatted XML string with proper DOCTYPE
        """
        try:
            # Processing instructions são criadas automaticamente pelo ET.tostring
            # Não é necessário remover manualmente
            
            # Criar a declaração XML e DOCTYPE
            xml_declaration = '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE xmeml>'
            
            # Converter para string sem formatação
            xml_content = ET.tostring(root, encoding='unicode')
            
            # Combinar declaração com o conteúdo XML sem usar minidom
            # para evitar declarações duplicadas ou formatação que possa quebrar o Premiere
            final_xml = f"{xml_declaration}{xml_content}"
            
            return final_xml
            
        except Exception as e:
            logger.error(f"Error formatting XML: {str(e)}")
            # Return basic XML without formatting (formato correto para o Premiere)
            # Garantindo que não há quebra de linha entre a declaração e a tag raiz
            return f'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE xmeml>{ET.tostring(root, encoding="unicode")}'

    
    def generate_cutting_guide(self, video_info: VideoInfo, 
                             editing_result: EditingResult) -> str:
        """
        Generate a human-readable cutting guide.
        
        Args:
            video_info: Video information
            editing_result: Editing results
            
        Returns:
            Cutting guide as formatted text
        """
        try:
            guide_lines = []
            
            # Header
            guide_lines.append("=" * 60)
            guide_lines.append("GUIA DE CORTES - EDITOR IA")
            guide_lines.append("=" * 60)
            guide_lines.append("")
            
            # Video information
            guide_lines.append("INFORMAÇÕES DO VÍDEO:")
            guide_lines.append(f"  Arquivo: {video_info.filename}")
            guide_lines.append(f"  Duração original: {video_info.duration:.1f}s")
            guide_lines.append(f"  Duração final: {editing_result.final_duration:.1f}s")
            guide_lines.append(f"  Taxa de compressão: {editing_result.compression_achieved:.1%}")
            guide_lines.append(f"  Resolução: {video_info.resolution}")
            guide_lines.append(f"  FPS: {video_info.fps}")
            guide_lines.append("")
            
            # Editing context
            guide_lines.append("CONTEXTO DE EDIÇÃO:")
            guide_lines.append(f"  Tipo de vídeo: {editing_result.context.video_type}")
            if editing_result.context.custom_instructions:
                guide_lines.append(f"  Instruções: {editing_result.context.custom_instructions}")
            guide_lines.append("")
            
            # Statistics
            guide_lines.append("ESTATÍSTICAS:")
            guide_lines.append(f"  Segmentos mantidos: {len(editing_result.selected_segments)}")
            guide_lines.append(f"  Total de decisões: {len(editing_result.decisions)}")
            guide_lines.append(f"  Pontuação de qualidade: {editing_result.quality_score:.1f}/10")
            guide_lines.append("")
            
            # Decisions
            guide_lines.append("DECISÕES DE EDIÇÃO:")
            guide_lines.append("-" * 40)
            
            for i, decision in enumerate(editing_result.decisions):
                if decision.decision_type == "keep":
                    guide_lines.append(f"#{i+1:03d} - MANTER")
                    guide_lines.append(f"  Função: {decision.function or 'N/A'}")
                    guide_lines.append(f"  Pontuação: {decision.score:.1f}/10")
                    guide_lines.append(f"  Confiança: {decision.confidence:.1%}")
                    guide_lines.append(f"  Motivo: {decision.reasoning}")
                    guide_lines.append("")
            
            # Warnings
            if editing_result.warnings:
                guide_lines.append("AVISOS:")
                for warning in editing_result.warnings:
                    guide_lines.append(f"  ⚠️ {warning}")
                guide_lines.append("")
            
            # Footer
            guide_lines.append("=" * 60)
            guide_lines.append("Gerado pelo Editor IA")
            guide_lines.append("=" * 60)
            
            return "\n".join(guide_lines)
            
        except Exception as e:
            logger.error(f"Error generating cutting guide: {str(e)}")
            return f"Erro ao gerar guia de cortes: {str(e)}"
    
    def validate_xml(self, xml_content: str) -> Dict[str, Any]:
        """
        Validate generated XML content.
        
        Args:
            xml_content: XML content to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Check root element
            if root.tag != "xmeml":
                result['errors'].append("Root element must be 'xmeml'")
                return result
            
            # Check version
            if root.get('version') != '4':
                result['warnings'].append("XMEML version should be '4'")
            
            # Check for required elements
            project = root.find('project')
            if project is None:
                result['errors'].append("Missing 'project' element")
                return result
            
            # Check for sequence
            sequence = root.find('.//sequence')
            if sequence is None:
                result['errors'].append("Missing 'sequence' element")
                return result
            
            # If we got here, XML is valid
            result['valid'] = True
            
        except ET.ParseError as e:
            result['errors'].append(f"XML parsing error: {str(e)}")
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def get_xml_stats(self, xml_content: str) -> Dict[str, Any]:
        """
        Get statistics about the generated XML.
        
        Args:
            xml_content: XML content to analyze
            
        Returns:
            Dictionary with XML statistics
        """
        stats = {
            'size_bytes': len(xml_content.encode('utf-8')),
            'size_kb': len(xml_content.encode('utf-8')) / 1024,
            'line_count': len(xml_content.splitlines()),
            'clip_count': 0,
            'track_count': 0,
            'sequence_count': 0
        }
        
        try:
            root = ET.fromstring(xml_content)
            
            # Count elements
            stats['clip_count'] = len(root.findall('.//clipitem'))
            stats['track_count'] = len(root.findall('.//track'))
            stats['sequence_count'] = len(root.findall('.//sequence'))
            
        except Exception as e:
            stats['error'] = str(e)
        
        return stats