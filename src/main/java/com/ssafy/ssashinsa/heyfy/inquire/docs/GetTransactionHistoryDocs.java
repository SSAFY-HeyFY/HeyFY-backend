package com.ssafy.ssashinsa.heyfy.inquire.docs;


import com.ssafy.ssashinsa.heyfy.account.dto.AccountNoDto;
import com.ssafy.ssashinsa.heyfy.inquire.dto.TransactionHistoryResponseRecDto; // 수정된 응답 DTO 임포트
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(
        summary = "계좌 거래 내역 조회",
        description = "연결된 계좌의 거래 내역을 조회합니다.",
        requestBody = @RequestBody(
                description = "거래 내역을 조회할 계좌번호",
                required = true,
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountNoDto.class),
                        examples = @ExampleObject(
                                name = "계좌번호 요청 예시",
                                value = "{\n  \"accountNo\": \"0014555423195469\"\n}"
                        )
                )
        )
)
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 거래 내역을 조회했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        // 스키마를 실제 반환되는 DTO로 변경
                        schema = @Schema(implementation = TransactionHistoryResponseRecDto.class),
                        // 제공된 성공 예시 추가
                        examples = @ExampleObject(
                                name = "거래 내역 조회 성공 예시",
                                value = "{\n  \"totalCount\": \"1\",\n  \"list\": [\n    {\n      \"transactionUniqueNo\": \"101438\",\n      \"transactionDate\": \"20250821\",\n      \"transactionTime\": \"225202\",\n      \"transactionType\": \"1\",\n      \"transactionTypeName\": \"입금\",\n      \"transactionAccountNo\": \"\",\n      \"transactionBalance\": \"1\",\n      \"transactionAfterBalance\": \"1\",\n      \"transactionSummary\": \"SSAFY 7335\",\n      \"transactionMemo\": \"\"\n    }\n  ]\n}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "400",
                description = "잘못된 요청: 계좌를 찾을 수 없음",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "계좌를 찾을 수 없음",
                                summary = "현재 유저의 DB에 요청된 계좌번호와 일치하는 계좌가 없는 경우",
                                value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"ACCOUNT_NOT_FOUND\", \"message\": \"유저와 연관된 계좌를 찾을 수 없습니다.\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류 또는 API 호출 실패",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "API 호출 실패",
                                value = "{\n  \"status\": 500,\n  \"httpError\": \"INTERNAL_SERVER_ERROR\",\n  \"errorCode\": \"API_CALL_FAILED\",\n  \"message\": \"신한 API 호출에 실패했습니다.\"\n}"
                        )
                )
        )
})
public @interface GetTransactionHistoryDocs {
}